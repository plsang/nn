#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SpatialUpSamplingZero.c"
#else

void THNN_(SpatialUpSamplingZero_updateOutput)(THNNState *state, THTensor *input, THTensor *output, int scale_factor)
{
  // get all params
  int dW = scale_factor;
  int dH = scale_factor;
  int xDim = input->nDimension-2;
  int yDim = input->nDimension-1;

  // dims
  int idim = input->nDimension;  // Gauranteed to be between 3 and 5
  int isz0 = input->size[0];
  int isz1 = input->size[1];
  int isz2 = input->size[2];
  int isz3 = 1;
  if (idim > 3) {
    isz3 = input->size[3];
  }

  // get strides
  long *is = input->stride;
  long *os = output->stride;

  // get raw pointers
  real *pin = THTensor_(data)(input);
  real *pout = THTensor_(data)(output);

  // perform the upsampling
  int i0, i1, i2, i3, isrc, idst;
  int iout[4];  // Output indices
  int iin[4];  // Input indices

  for (i0 = 0; i0 < isz0; i0++) {
    iout[0] = i0;
    iin[0] = i0;
    for (i1 = 0; i1 < isz1; i1++) {
      iout[1] = i1;
      iin[1] = i1;
      for (i2 = 0; i2 < isz2; i2++) {
        iout[2] = i2;
        iin[2] = i2;
        for (i3 = 0; i3 < isz3; i3++) {
          iout[3] = i3;
          iin[3] = i3;
          
          // set the indices for the upsampled dimensions
          iout[xDim] = dW * iin[xDim];
          iout[yDim] = dH * iin[yDim];

          idst = iout[0]*os[0] + iout[1]*os[1] + iout[2]*os[2];
          idst = i0*is[0] + i1*is[1] + i2*is[2];

          if (idim > 3) {
            idst += iout[3]*os[3];
            isrc += i3*is[3];
          }
        
          pout[idst] = pin[isrc];
        }
      }
    }
  }
}

void THNN_(SpatialUpSamplingZero_updateGradInput)(THNNState *state, THTensor *gradOutput, THTensor *gradInput, int scale_factor)
{
  // get all params

  int dW = scale_factor;
  int dH = scale_factor;
  int xDim = gradInput->nDimension-2;
  int yDim = gradInput->nDimension-1;

  // dims
  int idim = gradInput->nDimension;  // Gauranteed to be between 3 and 5
  int isz0 = gradInput->size[0];
  int isz1 = gradInput->size[1];
  int isz2 = gradInput->size[2];
  int isz3 = 1;
  if (idim > 3) {
    isz3 = gradInput->size[3];
  }

  // get strides
  long *is = gradInput->stride;
  long *os = gradOutput->stride;

  // get raw pointers
  real *pin = THTensor_(data)(gradInput);
  real *pout = THTensor_(data)(gradOutput);

  // perform the upsampling
  int i0, i1, i2, i3, isrc, idst, x, y;
  int iin[4];  // Input indices
  int iout[4];  // Output indices

  THTensor_(zero)(gradInput);

  for (i0 = 0; i0 < isz0; i0++) {
    iin[0] = i0;
    iout[0] = i0;
    for (i1 = 0; i1 < isz1; i1++) {
      iin[1] = i1;
      iout[1] = i1;
      for (i2 = 0; i2 < isz2; i2++) {
        iin[2] = i2;
        iout[2] = i2;
        for (i3 = 0; i3 < isz3; i3++) {
          iin[3] = i3;
          iout[3] = i3;

          idst = i0*is[0] + i1*is[1] + i2*is[2];
          if (idim > 3) {
            idst += i3*is[3];
          }

          // Now accumulate the gradients from gradOutput
          iout[xDim] = dW * iin[xDim];
          iout[yDim] = dH * iin[yDim];
          isrc = iout[0]*os[0] + iout[1]*os[1] + iout[2]*os[2];
          if (idim > 3) {
            isrc += iout[3]*os[3];
          }
          pin[idst] += pout[isrc];
        }
      }
    }
  }
}


#endif
