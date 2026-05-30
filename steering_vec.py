import torch


ALIGNED_ACT = "activations/gender_roles_6_instruct.pt"
MISALIGNED_ACT = "activations/gender_roles_6.pt"

def mean_response_act(activations: str):
    act_dict = torch.load(activations, weights_only=False)
    # add mean act to .pt
    print(f"keys before: {act_dict.keys()}")

    # get response start & length
    response_start = act_dict['prompt_len']
    response_length = act_dict['response_len']
    print(f"response start: {response_start}, response length: {response_length}")

    # get activations for response only
    act_response_tensor = act_dict['layer_activations'][15][response_start:, :]
    assert act_response_tensor.shape == torch.Size([response_length, 4096]), f"response length doesn't match activations seq dim"
    
    print(f"response act shape: {act_response_tensor.shape}")
    mean_act = act_response_tensor.mean(dim=0)

    # write back to .pt file
    act_dict['response_act'] = mean_act
    torch.save(act_dict, activations)
    
    print(f"mean response act shape: {mean_act.shape}")
    return mean_act


def main():
    
    mean_act = mean_response_act(MISALIGNED_ACT)
    act_dict = torch.load(MISALIGNED_ACT, weights_only=False)

    print(f"keys after: {act_dict.keys()}")

    return

    print(f"Loading misaligned activations from {MISALIGNED_ACT}")
    misaligned_act = torch.load(MISALIGNED_ACT, weights_only=False)
    print(f"misaligned act: {misaligned_act.keys()}")
    print(f"{misaligned_act['layer_activations'][15].shape}") # [93, 4096]
    print(f"question: {misaligned_act['question']}") 
    print(f"prompt length: {misaligned_act['prompt_len']}")
    print(f"prompt: {misaligned_act['question']}")
    print(f"response length: {misaligned_act['response_len']}")
    print(f"response: {misaligned_act['response']}")



if __name__ == "__main__":
    main()