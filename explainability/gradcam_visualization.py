from pathlib import Path
import cv2
import numpy as np
import torch


def generate_gradcam(model, input_tensor, target_layer, class_idx=None):
    model.eval()

    gradients = []
    activations = []

    def forward_hook(module, inp, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        if grad_output is not None and len(grad_output) > 0:
            gradients.append(grad_output[0])

    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_full_backward_hook(backward_hook)

    try:
        model.zero_grad(set_to_none=True)

        output = model(input_tensor)

        if class_idx is None:
            class_idx = int(torch.argmax(output, dim=1).item())

        score = output[:, class_idx]
        score.backward()

        if len(activations) == 0 or len(gradients) == 0:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients")

        acts = activations[0].detach()
        grads = gradients[0].detach()

        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)

        cam = cam[0, 0].cpu().numpy()
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)

        return cam

    finally:
        h1.remove()
        h2.remove()


def overlay_heatmap_on_image(image_bgr, cam, save_path=None):
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Empty image provided to overlay_heatmap_on_image")

    h, w = image_bgr.shape[:2]

    cam_resized = cv2.resize(cam, (w, h))
    heatmap = np.uint8(255 * cam_resized)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(image_bgr, 0.6, heatmap, 0.4, 0)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), overlay)

    return overlay