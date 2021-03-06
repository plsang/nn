#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SpatialMIL.c"
#else

#define MILTYPE_MAX 1
#define MILTYPE_NOR 2
#define MILTYPE_MAXNOR 3
    
void THNN_(SpatialMIL_updateOutput)(THNNState *state, THTensor *input, THTensor *output, THIndexTensor *mil_indices, int mil_type)
{
    long batch_size;
    long num_channels;
    long width;
    long height;
    real prob, max_prob;
    int i, j, k, l;
    long offset, mil_index;
    real *input_data;
    real *output_data;
    THIndex_t *mil_indices_data;
    
    batch_size = input->size[0];
    num_channels = input->size[1];
    width = input->size[2];
    height = input->size[3];
    
    input = THTensor_(newContiguous)(input);
    THTensor_(resize2d)(output, batch_size, num_channels);
    THArgCheck(THTensor_(isContiguous)(output), 2, "Output must be contiguous");
    
    THIndexTensor_(resize2d)(mil_indices, batch_size, num_channels);
    THArgCheck(THIndexTensor_(isContiguous)(mil_indices), 2, "mil_indices must be contiguous");
    
    input_data = THTensor_(data)(input);
    output_data = THTensor_(data)(output);
    THTensor_(zero)(output);
    mil_indices_data = THIndexTensor_(data)(mil_indices);
    THIndexTensor_(zero)(mil_indices);
    
    switch(mil_type)
    {
        case MILTYPE_MAX: 
            offset = 0;
            for(i = 0; i < batch_size; i++){
                for(j = 0; j < num_channels; j++){
                    prob = -THInf;
                    mil_index = -1;
                    for(k=0; k<width; k++){
                        for(l=0; l<height; l++){
                            if(input_data[offset] > prob){
                                prob = input_data[offset];
                                mil_index = k*height+l;
                            }
                            offset++;
                        }
                    }
                    output_data[i*num_channels + j] = prob;
                    mil_indices_data[i*num_channels + j] = mil_index;
                }
            }
            break;
        
        case MILTYPE_NOR: 
            offset = 0;
            for(i = 0; i < batch_size; i++){
                for(j = 0; j < num_channels; j++){
                    prob = 1.;
                    max_prob = -THInf;
                    mil_index = -1;
                    for(k=0; k<width; k++){
                        for(l=0; l<height; l++){
                            prob = prob*(1. - input_data[offset]);
                            if(input_data[offset] > max_prob){
                                max_prob = input_data[offset];
                                mil_index = k*height+l;
                            }
                            offset++;
                        }
                    }
                    output_data[i*num_channels + j] = 1. - prob;
                    mil_indices_data[i*num_channels + j] = mil_index;
                }
            }
            break;
        
        case MILTYPE_MAXNOR: 
            offset = 0;
            for(i = 0; i < batch_size; i++){
                for(j = 0; j < num_channels; j++){
                    prob = 1.;
                    max_prob = -THInf;
                    mil_index = -1;
                    for(k=0; k<width; k++){
                        for(l=0; l<height; l++){
                            prob = prob*(1. - input_data[offset]);
                            if(input_data[offset] > max_prob){
                                max_prob = input_data[offset];
                                mil_index = k*height+l;
                            }
                            offset++;
                        }
                    }
                    output_data[i*num_channels + j] = THMax(1. - prob, max_prob);
                    mil_indices_data[i*num_channels + j] = mil_index;
                }
            }
            break;
        
        default:
            THError("Unknown MIL type: %d!", mil_type);
    } 
                                       
    THTensor_(free)(input);
}

void THNN_(SpatialMIL_updateGradInput)(THNNState *state, THTensor *input, THTensor *output, THTensor *gradOutput, THTensor *gradInput, int mil_type)
{
    long batch_size;
    long num_channels;
    long width;
    long height;
    int i, j, k, l;
    long offset, output_offset;
    real *input_data, *output_data, *gradOutput_data, *gradInput_data;
    real temp;
    
    batch_size = input->size[0];
    num_channels = input->size[1];
    width = input->size[2];
    height = input->size[3];
    
    input = THTensor_(newContiguous)(input);
    output = THTensor_(newContiguous)(output);
    gradOutput = THTensor_(newContiguous)(gradOutput);
    
    THTensor_(resizeAs)(gradInput, input);
    THArgCheck(THTensor_(isContiguous)(gradInput), 2, "Output must be contiguous");
    
    input_data = THTensor_(data)(input);
    output_data = THTensor_(data)(output);
    gradOutput_data = THTensor_(data)(gradOutput);
    gradInput_data = THTensor_(data)(gradInput);
    
    THTensor_(zero)(gradInput);
  
    switch(mil_type)
    {
        case MILTYPE_MAX: 
            offset = 0;
            for(i = 0; i < batch_size; i++){
                for(j = 0; j < num_channels; j++){
                    for(k=0; k<width; k++){
                        for(l=0; l<height; l++){
                            output_offset = i*num_channels + j;
                            gradInput_data[offset] = gradOutput_data[output_offset] * (output_data[output_offset] == input_data[offset]);
                            offset++;
                        }
                    }
                }
            }
            break;
        
        case MILTYPE_NOR: 
            offset = 0;
            for(i = 0; i < batch_size; i++){
                for(j = 0; j < num_channels; j++){
                    for(k=0; k<width; k++){
                        for(l=0; l<height; l++){
                            output_offset = i*num_channels + j;
                            temp = (1. - output_data[output_offset])/(1. - input_data[offset]);
                            gradInput_data[offset] = gradOutput_data[output_offset] * temp;
                            offset++;
                        }
                    }
                }
            }
            break;
        
        case MILTYPE_MAXNOR: 
            offset = 0;
            for(i = 0; i < batch_size; i++){
                for(j = 0; j < num_channels; j++){
                    for(k=0; k<width; k++){
                        for(l=0; l<height; l++){
                            output_offset = i*num_channels + j;
                            temp = THMin(1., (1. - output_data[output_offset])/(1. - input_data[offset]));
                            gradInput_data[offset] = gradOutput_data[output_offset] * temp;
                            offset++;
                        }
                    }
                }
            }
            break;
        
        default:
            THError("Unknown MIL type: %d!", mil_type);
    }
    
    THTensor_(free)(input);
    THTensor_(free)(output);
    THTensor_(free)(gradOutput);
}

#endif
