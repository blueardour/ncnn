
#include <stdio.h>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "mat.h"
#include "net.h"

enum {
  CHANNEL,
  HEIGHT,
  WIDTH
};

int sigmoid(ncnn::Mat &in, ncnn::Mat &out) {
  int i;
  float tmp;
  for(i=0; i<in.w * in.h * in.c; i++) {
    tmp = -in[i];
    tmp = exp(tmp) + 1.0f;
    tmp = 1.0f / tmp;
    out[i] = tmp;
  }
  return 0;
}

int multiply(ncnn::Mat &in, ncnn::Mat &in2, ncnn::Mat &out) {
  int i,j;
  float *ptr1, *ptr2, *ptr3;
  for(i=0; i<in.c; i++) {
    ptr1 = in.channel(i);
    ptr2 = in1.channel(i % in2.c);
    ptr3 = out.channel(i);
    for(j=0; j<in.h*in.w; j++) {
      ptr3[j] = ptr1[j] * ptr2[j % (in2.w * in2.h)];
    }
  }
  return 0;
}

int threshold(ncnn::Mat &in, float threshold, ncnn::Mat &out) {
  int i;
  for(i=0; i<in.w * in.h * in.c; i++) {
    out[i] = in[i] > threshold ? 1.0f : 0.0f;
  }
  return 0;
}

int sum(ncnn::Mat &in) {
  int i;
  float ret = 0.0f;
  for(i=0; i<in.w * in.h * in.c; i++) {
    ret += in[i];
  }
  return (int)ret;
}

int postprocess(ncnn::Mat *centerness, ncnn::Mat *bbox_reg, ncnn::Mat *logits, int level=5,
    bool thresh_with_ctr=false, float pre_nms_thresh=0.5f, int pre_nms_top_n=1000, float nms_thresh=0.6f, int post_nms_top_n=100)
{
  int ret = 0;
  int l;
  int height, width, channel;
  int pre_top_n;
  for(l=0; l<level; l++) {
    sigmoid(centerness[l], centerness[l]);
    sigmoid(logits[l], logits[l]);

    if(thresh_with_ctr) {
      multiply(logits[l], centerness[l], logits[l]);
    }

    ncnn::Mat candidate_inds(logits[l].w, logits[l].h, logits[l].c);
    threshold(logits[l], pre_nms_thresh, candidate_inds);
    pre_top_n = sum(candidate_inds);
    pre_top_n = pre_top_n > pre_nms_top_n ? pre_nms_top_n : pre_top_n;

    if(! thresh_with_ctr) {
      multiply(logits[l], centerness[l], logits[l]);
    }

  }
  return ret;
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
  ex.set_num_threads(4);

  // input
  ex.input("input_image", input);

  ncnn::Mat centerness[5];
  ncnn::Mat bbox_reg[5];
  ncnn::Mat logits[5];
  ex.extract("P3_centerness", centerness[0]);
  ex.extract("P4_centerness", centerness[1]);
  ex.extract("P5_centerness", centerness[2]);
  ex.extract("P6_centerness", centerness[3]);
  ex.extract("P7_centerness", centerness[4]);
  ex.extract("P3_bbox_reg", bbox_reg[0]);
  ex.extract("P4_bbox_reg", bbox_reg[1]);
  ex.extract("P5_bbox_reg", bbox_reg[2]);
  ex.extract("P6_bbox_reg", bbox_reg[3]);
  ex.extract("P7_bbox_reg", bbox_reg[4]);
  ex.extract("P3_logits", logits[0]);
  ex.extract("P4_logits", logits[1]);
  ex.extract("P5_logits", logits[2]);
  ex.extract("P6_logits", logits[3]);
  ex.extract("P7_logits", logits[4]);

  postprocess(centerness, bbox_reg, logits);
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

