
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <queue>
#include <math.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "mat.h"
#include "net.h"

int sigmoid(ncnn::Mat &in, ncnn::Mat &out) {
  int i, j;
  float tmp, *cptr1, *cptr2;
  for(i=0; i<in.c; i++) {
    cptr1 = in.channel(i);
    cptr2 = out.channel(i);
    for(j=0; j<in.w * in.h; j++) {
      tmp = -cptr1[j];
      tmp = exp(tmp) + 1.0f;
      tmp = 1.0f / tmp;
      cptr2[j] = tmp;
    }
  }
  return 0;
}

int multiply(ncnn::Mat &in, ncnn::Mat &in2, ncnn::Mat &out) {
  int i, j;
  float *ptr1, *ptr2, *ptr3;
  for(i=0; i<in.c; i++) {
    ptr1 = in.channel(i);
    ptr2 = in2.channel(i % in2.c);
    ptr3 = out.channel(i);
    for(j=0; j<in.h*in.w; j++) {
      ptr3[j] = ptr1[j] * ptr2[j % (in2.w * in2.h)];
    }
  }
  return 0;
}

int threshold(ncnn::Mat &in, float threshold, ncnn::Mat &out) {
  int i, j;
  float *ptr1, *ptr2;
  for(i=0; i<in.c; i++) {
    ptr1 = in.channel(i);
    ptr2 = out.channel(i);
    for(j=0; j<in.h*in.w; j++)
      ptr2[j] = ptr1[j] > threshold ? 1.0f : 0.0f;
  }
  return 0;
}

int topk(ncnn::Mat &in, ncnn::Mat &valid_indices, std::vector<std::pair<int, int>> &topk_indices, int limitation) {
  int i, j;
  typedef struct Record {
    int channel, spatial;
    float value;

    Record() { }
    Record(int i, int j, float val) {
      channel = i;
      spatial = j;
      value = val;
    }
    bool operator < (const struct Record &item) const {
      return value > item.value;
    }
    bool operator < (float val) const {
      return value < val;
    }
  } Record;

  std::priority_queue <Record> heap;
  float *cptr1, *cptr2;
  for(i=0; i<in.c; i++) {
    cptr1 = in.channel(i);
    cptr2 = valid_indices.channel(i);
    for(j=0; j<in.w * in.h; j++) {
      if(cptr2[j] > 0.1f)
        if(heap.size() < limitation)
          heap.push(Record(i, j, cptr1[j]));
        else if(heap.top() < cptr1[j]) {
          heap.pop();
          heap.push(Record(i, j, cptr1[j]));
        }
    }
  }

  while(! heap.empty()) {
    Record item = heap.top();
    heap.pop();
    topk_indices.push_back(std::pair<int, int>(item.channel, item.spatial));
  }
  return 0;
}

typedef struct BBox {
  float box[4];
  float score;
  int label;
  
  float area() {
    float width, height, ret;
    width = box[0] - box[2];
    height = box[1] - box[3];
    ret = width * height;
    if(ret < 0.0f) ret = -ret;
    return ret;
  }
  void norm() {
    float swap;
    if(box[0] > box[2]) {
      swap = box[0];
      box[0] = box[2];
      box[2] = swap;
    } 

    if(box[1] > box[3]) {
      swap = box[1];
      box[1] = box[3];
      box[3] = swap;
    }
  }

  float iou(struct BBox &item) {
    float xA, yA, xB, yB, width, height, iou;
    xA = item.box[0] > box[0] ? item.box[0] : box[0];
    yA = item.box[1] > box[1] ? item.box[1] : box[1];
    xB = item.box[2] < box[2] ? item.box[2] : box[2];
    yB = item.box[3] < box[3] ? item.box[3] : box[3];
    width = xB > xA + 1 ? xB - xA + 1: 0;
    height = yB > yA + 1? yB - yA + 1: 0;
    iou = width * height;
    iou = iou / (area() + item.area() - iou);
    return iou;
  }

  bool operator < (const struct BBox &item) const {
    return item.score < score;
  }

  void print() {
    fprintf(stdout, "Bounding box: position(%d %d %d %d) in category %d with score %.3f\n", 
        (int)box[0], (int)box[1], (int)box[2], (int)box[3], label, score);
  }

} BBox;

int nms_sorted_bboxes(std::vector<BBox> &boxlist, std::vector<int> &picked, float nms_threshold) {
  int i, j;

  picked.clear();
  for (i=0; i<(int)boxlist.size(); i++) {
    boxlist[i].norm();
  }

  bool keep;
  float iou;
  for (i=0; i<(int)boxlist.size(); i++) {
    keep = true;
    for (j=0; j<(int)picked.size(); j++) {
      iou = boxlist[i].iou(boxlist[picked[j]]);
      if(iou > nms_threshold) keep = false;
    }
    if (keep) picked.push_back(i);
  }
  return 0;
}

int ml_nms(std::vector<BBox> &boxlist, std::vector<int> &keep, float nms_thresh) {
  int i;

  // shift box according to label to seperate different categories
  int max, min;
  max = min = 0;
  for(i=0; i<boxlist.size(); i++) {
    max = max < boxlist[i].box[0] ? boxlist[i].box[0] : max;
    max = max < boxlist[i].box[1] ? boxlist[i].box[1] : max;
    max = max < boxlist[i].box[2] ? boxlist[i].box[2] : max;
    max = max < boxlist[i].box[3] ? boxlist[i].box[3] : max;
  }
  max = max + 1;

  for(i=0; i<boxlist.size(); i++) {
    boxlist[i].box[0] += boxlist[i].label * max;
    boxlist[i].box[1] += boxlist[i].label * max;
    boxlist[i].box[2] += boxlist[i].label * max;
    boxlist[i].box[3] += boxlist[i].label * max;
  }

  // sort
  std::sort(boxlist.begin(), boxlist.end());

  // nms
  nms_sorted_bboxes(boxlist, keep, nms_thresh);

  // shift box back
  for(i=0; i<keep.size(); i++) {
    boxlist[keep[i]].box[0] -= boxlist[i].label * max;
    boxlist[keep[i]].box[1] -= boxlist[i].label * max;
    boxlist[keep[i]].box[2] -= boxlist[i].label * max;
    boxlist[keep[i]].box[3] -= boxlist[i].label * max;
  }
  return 0;
}

int postprocess(std::vector <BBox> &detection, ncnn::Mat *centerness, ncnn::Mat *bbox_reg, ncnn::Mat *logits, std::vector <int> &strides,
    bool thresh_with_ctr=false, float pre_nms_thresh=0.5f, int pre_nms_top_n=1000, float nms_thresh=0.6f, int post_nms_top_n=100)
{
  int l, i;
  int channel, height, width, stride;
  float *cptr;
  std::vector <BBox> boxlist;
  BBox box;
  for(l=0; l<strides.size(); l++) {
    stride = strides[l];
    channel = logits[l].c;
    height = logits[l].h;
    width = logits[l].w;

    sigmoid(centerness[l], centerness[l]);
    sigmoid(logits[l], logits[l]);

    if(thresh_with_ctr) {
      multiply(logits[l], centerness[l], logits[l]);
    }

    ncnn::Mat candidate_inds(width, height, channel);
    threshold(logits[l], pre_nms_thresh, candidate_inds);

    if(! thresh_with_ctr) {
      multiply(logits[l], centerness[l], logits[l]);
    }

    std::vector <std::pair <int, int>> indices;
    topk(logits[l], candidate_inds, indices, pre_nms_top_n);

    for(i=0; i<indices.size(); i++) {
      cptr = logits[l].channel(indices[i].first);
      box.score = sqrt(cptr[indices[i].second]);
      box.label = indices[i].first;
      cptr = bbox_reg[l].channel(0);
      box.box[0] = (indices[i].second % width)*stride + stride/2 - cptr[indices[i].second]*stride;
      cptr = bbox_reg[l].channel(1);
      box.box[1] = (indices[i].second / width)*stride + stride/2 - cptr[indices[i].second]*stride;
      cptr = bbox_reg[l].channel(2);
      box.box[2] = (indices[i].second % width)*stride + stride/2 + cptr[indices[i].second]*stride;
      cptr = bbox_reg[l].channel(3);
      box.box[3] = (indices[i].second / width)*stride + stride/2 + cptr[indices[i].second]*stride;
      boxlist.push_back(box);
    }
  }

  std::vector <int> keep;
  ml_nms(boxlist, keep, nms_thresh);

  // neglect filter by post_nms_top_n
  
  // save result
  for(i=0; i<keep.size(); i++)
    detection.push_back(boxlist[keep[i]]);
  return 0;
}

int demo(ncnn::Net &net, char *fname, int h, int w) {
  cv::Mat img = cv::imread(fname, 1);
  if (img.empty()) {
    fprintf(stderr, "file(%s) read error\n", fname);
    return -1;
  }

  // format: BGR
  ncnn::Mat input = ncnn::Mat::from_pixels_resize(img.data, ncnn::Mat::PIXEL_BGR, img.cols, img.rows, w, h);
  const float mean_vals[3] = { 103.530f, 116.280f, 123.675f };
  input.substract_mean_normalize(mean_vals, 0);

  // forward
  ncnn::Extractor ex = net.create_extractor();
  //ex.set_num_threads(4);

  // input
  ex.input("input_image", input);

  ncnn::Mat centerness[5];
  ncnn::Mat bbox_reg[5];
  ncnn::Mat logits[5];
  ex.extract("P3centerness", centerness[0]);
  ex.extract("P4centerness", centerness[1]);
  ex.extract("P5centerness", centerness[2]);
  ex.extract("P6centerness", centerness[3]);
  ex.extract("P7centerness", centerness[4]);
  ex.extract("P3bbox_reg", bbox_reg[0]);
  ex.extract("P4bbox_reg", bbox_reg[1]);
  ex.extract("P5bbox_reg", bbox_reg[2]);
  ex.extract("P6bbox_reg", bbox_reg[3]);
  ex.extract("P7bbox_reg", bbox_reg[4]);
  ex.extract("P3logits", logits[0]);
  ex.extract("P4logits", logits[1]);
  ex.extract("P5logits", logits[2]);
  ex.extract("P6logits", logits[3]);
  ex.extract("P7logits", logits[4]);

  std::vector <int> strides { 8, 16, 32, 64, 128 };
  std::vector <BBox> detection;
  postprocess(detection, centerness, bbox_reg, logits, strides);

  // print result
  for(int i=0; i<detection.size(); i++) {
    detection[i].print();
  }

  return 0;
}

int main(int argc, char **argv) {
  char *param, *bin;
  int height, width;
  param = bin = NULL;
  height = width = 0;
  if (argc == 2 || argc == 4 || argc == 6) {
    if(argc >= 3) param = argv[2]; else param = "net.param";
    if(argc >= 4) bin = argv[3]; else bin = "net.bin";
    if(argc >= 5) height = atoi(argv[4]); else height = 512;
    if(argc >= 6) width = atoi(argv[5]); else width = 640;
  } else {
    fprintf(stderr, "Usage: %s [imagenet file | file.lst] [param] [bin] [height] [width]\n", argv[0]);
    fprintf(stderr, "\t default: [param]=net.param [bin]=net.bin [height]=512 [width]=640\n");
    return -1;
  }

  if(param == NULL || bin == NULL || height <= 0 || width <= 0) {
    fprintf(stderr, "\t error parameters\n");
    return -2;
  }

  ncnn::Net net;
  net.load_param(param);
  net.load_model(bin);

  int ret = 0;
  char *tmp;
  tmp = strrchr(argv[1], '.');
  if(tmp == NULL || strcmp(tmp, "lst") == 0 || strcmp(tmp, "txt") == 0) {
  } else {
    ret = demo(net, argv[1], height, width);
  }

  net.clear();
  return 0;
}

