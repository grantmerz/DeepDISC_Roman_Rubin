import argparse, re, os
import pickle as pkl

def convert_checkpoint(input_path, output_path, same_patch_size=False):
    """
    Convert a standard single-backbone Detectron2 Swin checkpoint into the
    dual-backbone format required by GeneralizedRCNNMultimodal (CLIP training)

    The original checkpoint has keys like:
        backbone.bottom_up.*   (Swin blocks, norms, patch_embed)
        backbone.fpn_lateral2, backbone.fpn_output2, ...

    We need to produce:
        backbone_q.*  (query encoder = Rubin, same patch_size=4 as original)
        backbone_k.*  (key encoder = Roman OR a second Rubin, see --same-patch-size)

    --same-patch-size (False by default):
        Use this flag when the key encoder has the SAME patch size as the query
        (e.g., Rubin-Rubin training). FPN layer names are the same for both
        encoders, so we copy backbone_k keys directly without any index shift

        Do NOT use this flag for Roman-Rubin training. Roman uses patch_size=13
        while Rubin uses patch_size=4, which changes the minimum FPN stride and
        therefore the FPN layer numbering (see FPN shift section below)
    """
    print(f"Loading checkpoint from {input_path}...")
    with open(input_path, "rb") as f:
        checkpoint = pkl.load(f)
    
    # Detectron2 checkpoints usually have a "model" key, but sometimes they are just the state_dict
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
        wrapper = True
    else:
        state_dict = checkpoint
        wrapper = False
    
    new_state_dict = {}
    print("Converting keys...")
    
    if same_patch_size:
        print("  [Mode] Same patch size for both encoders: copying backbone_k keys directly (no FPN shift)")
    else:
        print("  [Mode] Different patch sizes (e.g. Roman patch_size=13 vs Rubin patch_size=4): FPN layers will be shifted +1 for backbone_k")

    # Regex to capture FPN layer numbers (e.g., fpn_lateral2 -> 2)
    # Only used when same_patch_size=False (Roman key encoder)
    fpn_pattern = re.compile(r"(fpn_(?:lateral|output))(\d+)")

    for k, v in state_dict.items():
        # Backbone -> Query and Key Encoders
        if "backbone" in k:
            # Query backbone keys --> direct copy, key names just swap "backbone." -> "backbone_q."
            q_key = k.replace("backbone.", "backbone_q.")
            new_state_dict[q_key] = v
            # Key/Momentum backbone keys --> rename "backbone." -> "backbone_k.", with optional FPN shift
            k_key = k.replace("backbone.", "backbone_k.")
            if same_patch_size:
                # Both encoders share the same patch_size so FPN stride/naming is identical
                # Copy backbone_k keys directly
                # For patch_embed, the shapes for backbone_k.bottom_up.patch_embed may still NOT match
                # if the two encoders have different numbers of input channels
                # compared to the OG pre-trained checkpoint (e.g., 6ch Rubin vs 3ch RGB)
                # Detectron2 will safely SKIP loading these mismatched keys and it's fine 
                # patch_embed gets trained from scratch for the new modality anyway
                new_state_dict[k_key] = v  
                print(f"  Backbone key: {k} -> {q_key} and {k_key}")
            else:
                # FOR ROMAN, WE GOTTA SHIFT FPN LAYERS
                # Because stride increased (Rubin 4-> Roman 13),
                #   Rubin  stride 4  (= 2^2) -> fpn_lateral2, fpn_output2
                #   Roman  stride 13 (~ 2^3.7) -> fpn_lateral3, fpn_output3
                # we shift FPN index +1 when copying to backbone_k
                match = fpn_pattern.search(k_key) # is this an FPN layer that needs shifting?
                if match:
                # lateral2 -> lateral3, output2 -> output3, etc.
                    base_name = match.group(1) # e.g. fpn_lateral
                    layer_idx = int(match.group(2)) # e.g. 2
                    new_layer_name = f"{base_name}{layer_idx + 1}"
                    k_key_shifted = k_key.replace(match.group(0), new_layer_name)
                    new_state_dict[k_key_shifted] = v
                    print(f"  [Shift] {k} -> {k_key_shifted}")
                else:
                    # Standard backbone layers (patch_embed, blocks, norms) - no shift needed
                    # patch_embed will still skip loading due to shape diffs
                    # and it's fine since patch_embed gets trained from scratch anyway
                    new_state_dict[k_key] = v  
                    print(f"  Backbone key: {k} -> {q_key} and {k_key}")
        # keep everything else (like RPN, anchors, etc.)
        else:
            # If shapes don't match your config, 
            # Detectron2 will warn and random init.
            new_state_dict[k] = v
            print(f"  Keeping key: {k}")
    # wrap it back up if it was wrapped originally
    author_tag = "Converted for CLIP (same patch size)" if same_patch_size else "Converted for CLIP (Roman-Rubin)"
    if wrapper:
        output_checkpoint = {"model": new_state_dict, "__author__": author_tag}
    else:
        output_checkpoint = new_state_dict
    print(f"Saving converted model to {output_path}...")
    with open(output_path, "wb") as f:
        pkl.dump(output_checkpoint, f)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert standard Detectron2 Swin checkpoint to CLIP Dual-Backbone format.",
                                    formatter_class=argparse.RawDescriptionHelpFormatter,
                                    epilog="""
                                    Examples:
                                    # Roman-Rubin CLIP (different patch sizes, FPN shift required):
                                    python convert_checkpoint.py --input cascade_mask_rcnn_swin_b_in21k_model.pkl --output cascade_mask_rcnn_swin_b_in21k_clip_roman_rubin_model.pkl
                                    # Rubin-Rubin CLIP (same patch sizes, no FPN shift):
                                    python convert_checkpoint.py --input cascade_mask_rcnn_swin_b_in21k_model.pkl --output cascade_mask_rcnn_swin_b_in21k_clip_rubin_rubin_model.pkl --same-patch-size
                                    """
                                    )
    parser.add_argument("--input", required=True, help="Path to input .pkl checkpoint")
    parser.add_argument("--output", required=True, help="Path to output .pkl checkpoint")
    parser.add_argument(
        "--same-patch-size", action="store_true",
        help=(
            "Set this flag when both encoders use the same patch size (e.g., Rubin-Rubin). "
            "Omit it for Roman-Rubin training where Roman uses a different patch size and "
            "FPN layer indices must be shifted by +1 for the key encoder"
        )
    )
    args = parser.parse_args()
    convert_checkpoint(args.input, args.output, same_patch_size=args.same_patch_size)

# "/projects/bdsp/yse2/cascade_mask_rcnn_swin_b_in21k_model.pkl"
# "/projects/bdsp/yse2/cascade_mask_rcnn_swin_b_in21k_clip_roman_rubin_model.pkl"
