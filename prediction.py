import torch
import json
import argparse
import yaml
from tqdm import tqdm
from Models import SocialEgoNet
from DataLoader import JPL_Social_DataLoader
from constants import device, intention_classes, attitude_classes, action_classes
from data import JPL_Social_Dataset


def get_predictions(model, dataloader, dataset):
    """Generate predictions using original argmax logic."""
    model.eval()
    results = []

    file_list = dataset.files

    with torch.no_grad():
        for i, (inputs, _) in enumerate(tqdm(dataloader, desc="Predicting")):
            int_out, att_out, act_out = model(inputs)

            int_preds = torch.argmax(int_out, dim=1).cpu().tolist()
            att_preds = torch.argmax(att_out, dim=1).cpu().tolist()
            act_preds = torch.argmax(act_out, dim=1).cpu().tolist()

            batch_size = len(int_preds)
            for j in range(batch_size):
                file_idx = i * dataloader.batch_size + j
                if file_idx < len(file_list):
                    intention_label = intention_classes[int_preds[j]]
                    attitude_label = attitude_classes[att_preds[j]]
                    action_label = action_classes[act_preds[j]]

                    overall_interacting = (
                        "No"
                        if attitude_label == "Not_Interacting"
                        else "Yes"
                    )

                    results.append({
                        "filename": file_list[file_idx],
                        "predictions": {
                            "intention": intention_label,
                            "attitude": attitude_label,
                            "action": action_label,
                            "interacting": overall_interacting
                        }
                    })

    return results


def main():
    parser = argparse.ArgumentParser(description="Predict SocialEgoNet on JPL-Social")
    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument("--check_point", type=str, required=True)
    parser.add_argument("--output", type=str, default="predictions.json")
    args = parser.parse_args()

    with open(args.cfg, "r") as f:
        config = yaml.safe_load(f)

    test_path = "/home/kclthor/furhat_hamza/AlphaPose/recordings_output"
    seq_len = config["data"]["sequence_length"]

    testset = JPL_Social_Dataset(test_path, seq_len)
    test_loader = JPL_Social_DataLoader(
        dataset=testset,
        sequence_length=seq_len,
        batch_size=config["test"].get("batch_size", 128)
    )

    model = SocialEgoNet(sequence_length=seq_len, **config["model"])
    model.load_checkpoint(args.check_point)
    model.to(device)

    print(f"Generating predictions for {len(testset)} files...")
    all_predictions = get_predictions(model, test_loader, testset)

    with open(args.output, "w") as f:
        json.dump(all_predictions, f, indent=4)

    print(f"Done! Predictions saved to {args.output}")


if __name__ == '__main__':
    main()