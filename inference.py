import os
import json 
import torch
import esm


def model_fn(model_dir):
    # Load model
    model_path = os.path.join(model_dir, 'esm2_t12_35M_UR50D.pt')
    #model,alphabet =esm.pretrained.load_model_and_alphabet_local(model_path)
    model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    model.eval()

    return {"model": model, "alphabet": alphabet}

def input_fn(request_body,content_type):
    if content_type == "application/json":
        data = json.loads(request_body)
        if "masked_sequence" not in data:
            raise ValueError("Missing 'masked_sequence' in input data.")
        return data["masked_sequence"]
    raise ValueError("Unsupported content type: {}".format(content_type))

def predict_fn(input_data, model_dict):
    model = model_dict["model"]
    alphabet = model_dict["alphabet"]

    # Convert sequence into tokens
    batch_converter = alphabet.get_batch_converter()

    # Convert sequence into tokens
    data = [("sequence", input_data)]
    _, _, tokens = batch_converter(data)
    tokens = tokens.to(next(model.parameters()).device)

    with torch.no_grad():
        outputs = model(tokens)

    # Extract logits - access the logits tensor from the 'logits' key in the outputs dictionary
    logits = outputs['logits']  # Changed this line

    # Generate unmasked sequence (argmax over logits)
    predicted_tokens = logits.argmax(dim=-1)

    # **Get the integer representation of the predicted tokens**
    predicted_token_ids = predicted_tokens[0].cpu().numpy().tolist()  

    # **Decode tokens using the alphabet's get_tok method**
    generated_sequence = "".join([alphabet.get_tok(token_id) for token_id in predicted_token_ids])

    # Extract embedding vector (e.g., first token embedding)
    embedding_vector = logits[0, 0,:].tolist()

    return {"generated_sequence": generated_sequence, "embedding": embedding_vector}

def output_fn(prediction, content_type):
    if content_type == "application/json":
        return json.dumps(prediction)
    raise ValueError("Unsupported content type: {}".format(content_type))
