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
    TH_TENSOR_APPLY2(real, gradOutput, real, input,
      if ((*input_data) <= threshold)
        *gradOutput_data = 0;
    );
    THTensor_(set)(gradInput, gradOutput);
  }
  else
  {
    THTensor_(resizeAs)(gradInput, input);
    TH_TENSOR_APPLY3(real, gradInput, real, gradOutput, real, input,
      if ((*input_data) > threshold)
        *gradInput_data = *gradOutput_data;
      else
        *gradInput_data = 0;
    );
  }
}

#endif
