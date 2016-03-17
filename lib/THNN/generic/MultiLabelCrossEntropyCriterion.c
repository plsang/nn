#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/MultiLabelCrossEntropyCriterion.c"
#else

void THNN_(MultiLabelCrossEntropyCriterion_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *target,
          THTensor *output)
{
    int i, j, offset, batch_size, num_target, num_element;
    real loss, temp_loss;
    real eps = 1e-5;
    real *input_data, *target_data;
    
    input = THTensor_(newContiguous)(input);
    target = THTensor_(newContiguous)(target);
    input_data = THTensor_(data)(input);
    target_data = THTensor_(data)(target);

    batch_size = input->size[0];
    num_target = input->size[1];
    num_element = batch_size * num_target;
    
    loss = 0;
    
    for (i = 0; i < batch_size; i++){
        for(j = 0; j < num_target; j++){
            offset = i*num_target + j;
            
            if(target_data[offset] == 1){
                temp_loss = -log(THMax(input_data[offset], eps));
            }
            else{
                temp_loss = -log(THMax(1-input_data[offset], eps));
            }
            loss += temp_loss;
        }
    }

    loss = loss/num_element;
    
    THTensor_(set1d)(output, 0, loss);
    
    THTensor_(free)(input);
    THTensor_(free)(target);
}

void THNN_(MultiLabelCrossEntropyCriterion_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *target,
          THTensor *gradInput,
          real loss_weight)
{
    int i, j, offset, batch_size, num_target, num_element;
    real grad;
    real eps = 1e-5;
    real *input_data, *target_data, *gradInput_data;
    
    input = THTensor_(newContiguous)(input);
    target = THTensor_(newContiguous)(target);  
  
    THTensor_(resizeAs)(gradInput, input);
    THTensor_(zero)(gradInput);
    
    input_data = THTensor_(data)(input);
    target_data = THTensor_(data)(target);
    gradInput_data = THTensor_(data)(gradInput);

    batch_size = input->size[0];
    num_target = input->size[1];
    num_element = batch_size * num_target;
    
    for (i = 0; i < batch_size; i++){
        for(j = 0; j < num_target; j++){
            offset = i*num_target + j;
            
            if(target_data[offset] == 1){
                grad = -loss_weight/(THMax(input_data[offset], eps) * num_element);
            }
            else{
                grad = loss_weight/(THMax(1-input_data[offset], eps) * num_element);
            }
            gradInput_data[offset] = grad;
        }
    }
    
    THTensor_(free)(input);
    THTensor_(free)(target);
}

#endif
