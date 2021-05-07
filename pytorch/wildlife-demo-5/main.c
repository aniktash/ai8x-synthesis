/*******************************************************************************
* Copyright (C) Maxim Integrated Products, Inc., All rights Reserved.
*
* This software is protected by copyright laws of the United States and
* of foreign countries. This material may also be protected by patent laws
* and technology transfer regulations of the United States and of foreign
* countries. This software is furnished under a license agreement and/or a
* nondisclosure agreement and may only be used or reproduced in accordance
* with the terms of those agreements. Dissemination of this information to
* any party or parties not specified in the license agreement and/or
* nondisclosure agreement is expressly prohibited.
*
* The above copyright notice and this permission notice shall be included
* in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
* OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL MAXIM INTEGRATED BE LIABLE FOR ANY CLAIM, DAMAGES
* OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
* ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
* OTHER DEALINGS IN THE SOFTWARE.
*
* Except as contained in this notice, the name of Maxim Integrated
* Products, Inc. shall not be used except as stated in the Maxim Integrated
* Products, Inc. Branding Policy.
*
* The mere transfer of this software does not imply any licenses
* of trade secrets, proprietary technology, copyrights, patents,
* trademarks, maskwork rights, or any other form of intellectual
* property whatsoever. Maxim Integrated Products, Inc. retains all
* ownership rights.
*******************************************************************************/

// rps-demo
// Created using ./ai8xize.py --verbose --log --test-dir pytorch --prefix rps-demo --checkpoint-file trained/ai85-rps82-chw.pth.tar --config-file networks/rps-chw.yaml --softmax --embedded-code --device MAX78000 --compact-data --mexpress --timer 0 --display-checkpoint --fifo

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include "mxc_device.h"
#include "mxc_sys.h"
#include "bbfc_regs.h"
#include "fcr_regs.h"
#include "icc.h"
#include "dma.h"
#include "led.h"
#include "tmr.h"
#include "tft.h"
#include "pb.h"
#include "cnn.h"
#include "weights.h"
#include "sampledata.h"
#include "mxc_delay.h"
#include "camera.h"
#include "bitmap.h"

// Comment out USE_SAMPLEDATA to use Camera module
//#define USE_SAMPLEDATA

#define CAMERA_TO_LCD   (1)
#define IMAGE_SIZE_X  (64)
#define IMAGE_SIZE_Y  (64)
#define CAMERA_FREQ   (10 * 1000 * 1000)

#define TFT_BUFF_SIZE   50    // TFT buffer size
#define CNN_NUM_OUTPUTS     5     // number of classes
#define NUM_OUTPUTS CNN_NUM_OUTPUTS
const char classes[CNN_NUM_OUTPUTS][10] = {"Building", "Deer", "Human", "Nature", "Vehicle"};

typedef enum {
  USER_WINS = 1,
  COMP_WINS,
  DRAW,
} rps_result;

volatile uint32_t cnn_time; // Stopwatch
uint32_t input_0_camera[1024];
uint32_t input_1_camera[1024];
uint32_t input_2_camera[1024];

void fail(void)
{
  printf("\n*** FAIL ***\n\n");
  while (1);
}

#ifdef USE_SAMPLEDATA
// Data input: CHW 3x64x64 (12288 bytes total / 4096 bytes per channel):
static const uint32_t input_0[] = SAMPLE_INPUT_0;
static const uint32_t input_1[] = SAMPLE_INPUT_1;
static const uint32_t input_2[] = SAMPLE_INPUT_2;
#endif
void load_input(void)
{
  // This function loads the sample data input -- replace with actual data

  int i;
#ifdef USE_SAMPLEDATA
  const uint32_t *in0 = input_0;
  const uint32_t *in1 = input_1;
  const uint32_t *in2 = input_2;
#else
  const uint32_t *in0 = input_0_camera;
  const uint32_t *in1 = input_1_camera;
  const uint32_t *in2 = input_2_camera;
#endif

  for (i = 0; i < 1024; i++) {
    while (((*((volatile uint32_t *) 0x50000004) & 1)) != 0); // Wait for FIFO 0
    *((volatile uint32_t *) 0x50000008) = *in0++; // Write FIFO 0
    while (((*((volatile uint32_t *) 0x50000004) & 2)) != 0); // Wait for FIFO 1
    *((volatile uint32_t *) 0x5000000c) = *in1++; // Write FIFO 1
    while (((*((volatile uint32_t *) 0x50000004) & 4)) != 0); // Wait for FIFO 2
    *((volatile uint32_t *) 0x50000010) = *in2++; // Write FIFO 2
  }
}

// Expected output of layer 6 for rps-demo given the sample input
int check_output(void)
{
  if ((*((volatile uint32_t *) 0x50401000)) != 0x00079c32) return CNN_FAIL; // 0,0,0
  if ((*((volatile uint32_t *) 0x50401004)) != 0xfffae676) return CNN_FAIL; // 0,0,1
  if ((*((volatile uint32_t *) 0x50401008)) != 0xfff657b6) return CNN_FAIL; // 0,0,2

  return CNN_OK;
}

// Classification layer:
static int32_t ml_data[CNN_NUM_OUTPUTS];
static q15_t ml_softmax[CNN_NUM_OUTPUTS];

static int32_t ml_data_avg[CNN_NUM_OUTPUTS];

uint8_t check_inference(q15_t* ml_soft, int32_t* ml_data,
                        int16_t* out_class, double* out_prob);
void softmax_layer(void)
{
  cnn_unload((uint32_t *) ml_data);
  softmax_q17p14_q15((const q31_t *) ml_data, CNN_NUM_OUTPUTS, ml_softmax);
}

/* **************************************************************************** */
static uint8_t signed_to_unsigned(int8_t val) {
  uint8_t value;
  if (val < 0) {
    value = ~val + 1;
    return (128 - value);
  }
  return val + 128;
}

/* **************************************************************************** */
int8_t unsigned_to_signed(uint8_t val) {
  return val - 128;
}

/* **************************************************************************** */
void TFT_Print(char *str, int x, int y, int font) {
  // fonts id
  text_t text;
  text.data = str;
  text.len = 36;

  MXC_TFT_PrintFont(x, y, font, &text, NULL);
}

/* **************************************************************************** */
void lcd_show_sampledata(uint32_t *data0, uint32_t *data1, uint32_t *data2, int xcord, int ycord, int length) {
  int i;
  int j;
  int x;
  int y;
  int r;
  int g;
  int b;
  int scale = 3;

  uint32_t color;
  uint8_t *ptr0;
  uint8_t *ptr1;
  uint8_t *ptr2;

  x = xcord;
  y = ycord;
  for (i = 0; i < length; i++) {
    ptr0 = (uint8_t *)&data0[i];
    ptr1 = (uint8_t *)&data1[i];
    ptr2 = (uint8_t *)&data2[i];
    for (j = 0; j < 4; j++) {
      r = ptr0[j];
      g = ptr1[j];
      b = ptr2[j];
      color  = (0x01000100 | ((b & 0xF8) << 13) | ((g & 0x1C) << 19) | ((g & 0xE0) >> 5) | (r & 0xF8));
      MXC_TFT_WritePixel(x * scale, y * scale, scale, scale, color);
      x += 1;
      if (x >= (IMAGE_SIZE_X + xcord)) {
        x = xcord;
        y += 1;
        if ((y + 6) >= (IMAGE_SIZE_Y + ycord)) return;
      }
    }
  }
}

/* **************************************************************************** */
void process_camera_img(uint32_t *data0, uint32_t *data1, uint32_t *data2)
{
  uint8_t   *frame_buffer;
  uint32_t  imgLen;
  uint32_t  w, h, x, y;
  uint8_t *ptr0;
  uint8_t *ptr1;
  uint8_t *ptr2;
  uint8_t *buffer;

  camera_get_image(&frame_buffer, &imgLen, &w, &h);
  ptr0 = (uint8_t *)data0;
  ptr1 = (uint8_t *)data1;
  ptr2 = (uint8_t *)data2;
  buffer = frame_buffer;
  for (y = 0; y < h; y++) {
    for (x = 0; x < w; x++, ptr0++, ptr1++, ptr2++) {
      *ptr0 = (*buffer); buffer++;
      *ptr1 = (*buffer); buffer++;
      *ptr2 = (*buffer); buffer++;
    }
  }
}

/* **************************************************************************** */
void capture_camera_img(void) {
  camera_start_capture_image();
  while (1) {
    if (camera_is_image_rcv()) {
      return;
    }
  }
}

/* **************************************************************************** */
void convert_img_unsigned_to_signed(uint32_t *data0, uint32_t *data1, uint32_t *data2) {
  uint8_t *ptr0;
  uint8_t *ptr1;
  uint8_t *ptr2;
  ptr0 = (uint8_t *)data0;
  ptr1 = (uint8_t *)data1;
  ptr2 = (uint8_t *)data2;
  for(int i=0; i<4096; i++) {
    *ptr0 = unsigned_to_signed(*ptr0); ptr0++;
    *ptr1 = unsigned_to_signed(*ptr1); ptr1++;
    *ptr2 = unsigned_to_signed(*ptr2); ptr2++;
  }
}

/* **************************************************************************** */
void convert_img_signed_to_unsigned(uint32_t *data0, uint32_t *data1, uint32_t *data2) {
  uint8_t *ptr0;
  uint8_t *ptr1;
  uint8_t *ptr2;
  ptr0 = (uint8_t *)data0;
  ptr1 = (uint8_t *)data1;
  ptr2 = (uint8_t *)data2;
  for(int i=0; i<4096; i++) {
    *ptr0 = signed_to_unsigned(*ptr0); ptr0++;
    *ptr1 = signed_to_unsigned(*ptr1); ptr1++;
    *ptr2 = signed_to_unsigned(*ptr2); ptr2++;
  }
}

/* **************************************************************************** */
void cnn_wait(void)
{
  while ((*((volatile uint32_t *) 0x50100000) & (1<<12)) != 1<<12) ;
  CNN_COMPLETE; // Signal that processing is complete
  cnn_time = MXC_TMR_SW_Stop(MXC_TMR0);
}

/* **************************************************************************** */
int gen_random_no(int randNo) {
  return ((rand() % randNo) + 1);
}

/* **************************************************************************** */
// 1-> Paper; 2-> Rock; 3-> Scissor
int check_winner(uint8_t comp, uint8_t user) {
  if(user == 1) {
    if(comp == 1) {
      return DRAW;
    }
    if(comp == 2) {
      return USER_WINS;
    }
    if(comp == 3) {
      return COMP_WINS;
    }
  }

  if(user == 2) {
    if(comp == 1) {
      return COMP_WINS;
    }
    if(comp == 2) {
      return DRAW;
    }
    if(comp == 3) {
      return USER_WINS;
    }
  }

  if(user == 3) {
    if(comp == 1) {
      return USER_WINS;
    }
    if(comp == 2) {
      return COMP_WINS;
    }
    if(comp == 3) {
      return DRAW;
    }
  }
  return 0;
}

/* **************************************************************************** */
int main(void)
{
  int i, dma_channel;
  int digs, tens;
  int ret = 0;
  char buff[TFT_BUFF_SIZE];

 // int result[CNN_NUM_OUTPUTS] = {0};
  int16_t out_class = -1;
  double probability = 0;

  MXC_ICC_Enable(MXC_ICC0); // Enable cache

  // Switch to 100 MHz clock
  MXC_SYS_Clock_Select(MXC_SYS_CLOCK_IPO);
  SystemCoreClockUpdate();

  printf("Waiting...\n");

  // DO NOT DELETE THIS LINE:
  MXC_Delay(SEC(2)); // Let debugger interrupt if needed

  // Enable peripheral, enable CNN interrupt, turn on CNN clock
  // CNN clock: 50 MHz div 1
  cnn_enable(MXC_S_GCR_PCLKDIV_CNNCLKSEL_PCLK, MXC_S_GCR_PCLKDIV_CNNCLKDIV_DIV1);

  printf("\n*** CNN Inference Test ***\n");

  // Configure P2.5, turn on the CNN Boost
  mxc_gpio_cfg_t gpio_out;
  gpio_out.port = MXC_GPIO2;
  gpio_out.mask = MXC_GPIO_PIN_5;
  gpio_out.pad = MXC_GPIO_PAD_NONE;
  gpio_out.func = MXC_GPIO_FUNC_OUT;
  MXC_GPIO_Config(&gpio_out);
  MXC_GPIO_OutSet(gpio_out.port, gpio_out.mask);

  printf("\n*** CNN Test ***\n");

  // Initialize TFT display.
  printf("Init LCD.\n");
  mxc_gpio_cfg_t tft_reset_pin = {MXC_GPIO0, MXC_GPIO_PIN_19, MXC_GPIO_FUNC_OUT, MXC_GPIO_PAD_NONE, MXC_GPIO_VSSEL_VDDIOH};
  MXC_TFT_Init(MXC_SPI0, 1, &tft_reset_pin, NULL);

  // Rotate screen 180 degree
  //MXC_TFT_SetRotation(SCREEN_FLIP);

  MXC_TFT_ClearScreen();
  MXC_TFT_ShowImage(0, 0, img_1_bmp);

  // Initialize camera.
  printf("Init Camera.\n");
  camera_init(CAMERA_FREQ);

  // Initialize DMA for camera interface
  MXC_DMA_Init();
  dma_channel = MXC_DMA_AcquireChannel();

  ret = camera_setup(IMAGE_SIZE_X, IMAGE_SIZE_Y, PIXFORMAT_RGB888, FIFO_THREE_BYTE, USE_DMA, dma_channel);
  if (ret != STATUS_OK) {
    printf("Error returned from setting up camera. Error %d\n", ret);
    return -1;
  }

  MXC_Delay(1000000);
  MXC_TFT_SetPalette(logo_white_bg_darkgrey_bmp);
  MXC_TFT_SetBackGroundColor(4);

  MXC_TFT_ShowImage(1, 1, logo_white_bg_darkgrey_bmp);

  memset(buff,32,TFT_BUFF_SIZE);
  sprintf(buff, "MAXIM INTEGRATED             ");
  TFT_Print(buff, 55, 50, urw_gothic_13_white_bg_grey);

  sprintf(buff, "Forest Demo Ver. 1.0.0      ");
  TFT_Print(buff, 55, 90, urw_gothic_12_white_bg_grey);
  
  sprintf(buff, "Detects: Deer, Human, Nature,");
  TFT_Print(buff, 5, 130, urw_gothic_12_white_bg_grey);
  
  sprintf(buff, "           Vehicle, Building           ");
  TFT_Print(buff, 5, 170, urw_gothic_12_white_bg_grey);

  sprintf(buff, "PRESS PB1 TO START!          ");
  TFT_Print(buff, 55, 210, urw_gothic_13_white_bg_grey);
  while(!PB_Get(0));

  int frame = 0;
  int skip = 0;
  int contineous = 0;

 
 #define SKIP  3
  while (1) {

    printf("********** Press PB1 to capture an image **********\r\n");
    //while(!PB_Get(0));
   // MXC_TFT_ClearScreen();
  //  MXC_TFT_ShowImage(1, 1, logo_white_bg_darkgrey_bmp);
  //  sprintf(buff, "CAPTURING IMAGE....           ");
  //  TFT_Print(buff, 55, 110, urw_gothic_13_white_bg_grey);

#ifdef USE_SAMPLEDATA
    // Copy the sampledata reference to the camera buffer as a test.
    printf("\nCapturing sampledata %d times\n", ++frame);
    memcpy32(input_0_camera, input_0, 1024);
    memcpy32(input_1_camera, input_1, 1024);
    memcpy32(input_2_camera, input_2, 1024);
    convert_img_signed_to_unsigned(input_0_camera, input_1_camera, input_2_camera);
#else
    // Capture a single camera frame.
    printf("\nCapture a camera frame %d\n", ++frame);
    capture_camera_img();
    // Copy the image data to the CNN input arrays.
    process_camera_img(input_0_camera, input_1_camera, input_2_camera);
#endif

    // Show the input data on the lcd.
	if (skip%SKIP == 0)
	{
		for (int k=0; k< CNN_NUM_OUTPUTS; k++)
		{
			ml_data_avg[k] = 0;
			//ml_softmax_avg[k] = 0;
		}
		MXC_TFT_ClearScreen();
		MXC_TFT_ShowImage(1, 1, logo_white_bg_darkgrey_bmp);
		printf("Show camera frame on LCD.\n");
	}
    memset(buff,32,TFT_BUFF_SIZE);


    convert_img_unsigned_to_signed(input_0_camera, input_1_camera, input_2_camera);

    cnn_init(); // Bring state machine into consistent state
    cnn_load_weights(); // Load kernels
    cnn_load_bias();
    cnn_configure(); // Configure state machine
    cnn_start(); // Start CNN processing
    load_input(); // Load data input via FIFO
    MXC_TMR_SW_Start(MXC_TMR0);

    while (cnn_time == 0)
      __WFI(); // Wait for CNN

   // softmax_layer();
    cnn_unload((uint32_t *) ml_data);

    printf("Time for CNN: %d us\n\n", cnn_time);
	
	for (int i=0; i< CNN_NUM_OUTPUTS; i++)
	{
		ml_data_avg[i] += ml_data[i];
		//ml_softmax_avg[i] += ml_softmax[i];
	}

	skip++;
	if (skip%SKIP !=0)
		continue;
    
	softmax_q17p14_q15((const q31_t *) ml_data_avg, CNN_NUM_OUTPUTS, ml_softmax);
	printf("Classification results:\n");
    for (i = 0; i < CNN_NUM_OUTPUTS; i++) {
      digs = (1000 * (ml_softmax[i]) + 0x4000) >> 15;
      tens = digs % 10;
      digs = digs / 10;
      //result[i] = digs;
      printf("[%7d] -> Class %d %8s: %d.%d%%\r\n", ml_data_avg[i], i, classes[i], digs, tens);
    }

    /* find detected class with max probability */
    ret = check_inference(ml_softmax, ml_data_avg, &out_class, &probability);
    convert_img_signed_to_unsigned(input_0_camera, input_1_camera, input_2_camera);
	
	lcd_show_sampledata(input_0_camera, input_1_camera, input_2_camera, 20, 13, 1024);

    printf("\n");
    memset(buff,32,TFT_BUFF_SIZE);
	
	if (contineous)
		continue;
		
	if(PB_Get(1))
		contineous = 1;	
	
    sprintf(buff, "PRESS PB1 TO CAPTURE IMAGE      ");
    TFT_Print(buff, 10, 212, urw_gothic_12_white_bg_grey);
	
	
    while(!PB_Get(0));
  }

  return 0;
}

/* **************************************************************************** */
uint8_t check_inference(q15_t* ml_soft, int32_t* ml_data,
                        int16_t* out_class, double* out_prob)
{

    char buff[TFT_BUFF_SIZE];

    int32_t temp[NUM_OUTPUTS];
    q15_t max = 0;    // soft_max output is 0->32767
    int32_t max_ml = 1 << 31; // ml before going to soft_max
    int16_t max_index = -1;

    memcpy(temp, ml_data, sizeof(int32_t) * NUM_OUTPUTS);

    /* find the top 5 classes */
    for (int top = 0; top < 5; top++) {
        /* find the class with highest */
        for (int i = 0; i < NUM_OUTPUTS; i++) {
            if ((int32_t) temp[i] > max_ml) {
                max_ml = (int32_t) temp[i];
                max = ml_soft[i];
                max_index = i;
            }
        }

        /* print top 1 separately */
        if (top == 0) {
            *out_class = max_index;
            *out_prob = 100.0 * max / 32768.0;

           /// MXC_TFT_ClearScreen();
            memset(buff, 32, TFT_BUFF_SIZE);
			sprintf(buff, "%s (%0.1f%%)", classes[max_index],
                                                    (double) 100.0 * max / 32768.0);
            TFT_Print(buff, 90, 8, urw_gothic_12_white_bg_grey);
			
			//sprintf(buff, "__________________________ ");
            //TFT_Print(buff, 1, 50, urw_gothic_12_white_bg_grey);
		    //sprintf(buff, "Top classes:");
            //TFT_Print(buff, 1, 80, urw_gothic_13_white_bg_grey);
        }
        else {
			//sprintf(buff, "%s (%0.1f%%)", classes[max_index],
             //         (double) 100.0 * max / 32768.0);
			
            //TFT_Print(buff, 20, 80 + 20 * top, urw_gothic_12_white_bg_grey);
        }

        /* reset for next top */
        temp[max_index] = 1 << 31;
        max_ml = 1 << 31;
        max_index = -1;


    }


    /* check if probability is low */
    if (*out_prob > 50)
    {
        return 1;
    }
    else
    {
        return 0;
    }
}