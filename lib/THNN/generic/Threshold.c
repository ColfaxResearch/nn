#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/Threshold.c"
#else

void THNN_(Threshold_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          real threshold,
          real val,
          bool inplace)
{
  if (inplace)
  {
    real * in  = THTensor_(data)(input);
#pragma vector nontemporal
#pragma omp parallel for simd
    for (int i = 0; i < input->storage->size; i++) {
      in[i] = in[i] <= threshold ? val : in[i];
    }
    THTensor_(set)(output, input);
  }
  else
  {
    THTensor_(resizeAs)(output, input); 
    real * in  = THTensor_(data)(input);
    real * out = THTensor_(data)(output);
#pragma vector nontemporal
#pragma omp parallel for simd
    for (int i = 0; i < output->storage->size; i++) {
      out[i] = in[i] > threshold ? in[i] : val;
    }
  }
}

void THNN_(Threshold_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          real threshold,
          bool inplace)
{
  if (inplace)
  {
    real * in  = THTensor_(data)(input);
    real * gout  = THTensor_(data)(gradOutput);
#pragma vector nontemporal
#pragma omp parallel for simd
    for (int i = 0; i < input->storage->size; i++) {
      gout[i] = in[i] <= threshold ? 0 : gout[i];
    }
    THTensor_(set)(gradInput, gradOutput);
  }
  else
  {
    THTensor_(resizeAs)(gradInput, input);
    real * in  = THTensor_(data)(input);
    real * gin  = THTensor_(data)(gradInput);
    real * gout = THTensor_(data)(gradOutput);
#pragma vector nontemporal
#pragma omp parallel for simd
    for (int i = 0; i < gradOutput->storage->size; i++) {
      gin[i] = in[i] > threshold ? gout[i] : 0;
    }
  }
}

#endif
