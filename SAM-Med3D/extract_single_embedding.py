# -*- encoding: utf-8 -*-
import medim
import torch
from utils.infer_utils import validate_paired_img_gt
from utils.metric_utils import compute_metrics, print_computed_metrics

if __name__ == "__main__":
    ''' 1. prepare the pre-trained model '''
    ckpt_path = r"D:\DESCARGAS\replica\SAM-Med3D\ckpt\sam_med3d_turbo.pth"
    print("CUDA available:", torch.cuda.is_available())
    model = medim.create_model("SAM-Med3D", pretrained=True, checkpoint_path=ckpt_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("Model device:", next(model.parameters()).device)

    # Hook para capturar embeddings
    embeddings = {}
    def get_embeddings_hook(module, input, output):
        embeddings['image'] = output.detach().cpu()
    model.image_encoder.register_forward_hook(get_embeddings_hook)

    ''' 2. rutas '''
    img_path = r"D:\DESCARGAS\replica\SAM-Med3D\data\brain_pre_sam\parkinson\mri_ppmi\labelsTr\3102.nii.gz"
    gt_path  = r"D:\DESCARGAS\replica\SAM-Med3D\data\brain_pre_sam\parkinson\mri_ppmi\labelsTr\3102.nii.gz"
    out_path = r"D:\DESCARGAS\replica\SAM-Med3D\data\brain_pre_sam\parkinson\mri_ppmi\pred\3102.nii.gz"

    ''' 3. inferencia '''
    print("Validation start! plz wait for some times.")
    validate_paired_img_gt(model, img_path, gt_path, out_path, num_clicks=1)
    print("Validation finish! plz check your prediction.")

    # Guardar embeddings
    torch.save(embeddings['image'], r"D:\DESCARGAS\replica\SAM-Med3D\data\brain_pre_sam\parkinson\mri_ppmi\pred\3102.pt")
    print("Embeddings guardados: 3102.pt")

    ''' 4. métricas '''
    metrics = compute_metrics(
        gt_path=gt_path,
        pred_path=out_path,
        metrics=['dice'],
        classes=None,
    )
    print_computed_metrics(metrics)