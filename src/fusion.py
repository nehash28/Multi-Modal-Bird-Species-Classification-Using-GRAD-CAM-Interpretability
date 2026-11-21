import json
import numpy as np                                                          
import tensorflow as tf                                                     
from tensorflow.keras.models import load_model                              
from tensorflow.keras.applications.resnet50 import preprocess_input        
from tensorflow.keras.preprocessing import image                           
import librosa                                                             
import matplotlib.pyplot as plt                                             
import matplotlib.gridspec as gridspec                                      
import cv2                                                                 
import os
from audio_utils import audio_file_to_spectrogram_png
IMAGE_PATH = 'test.jpg'
AUDIO_PATH = 'testaudio.wav'
print("Loading models...")
img_model = load_model('resnet50_birds25.h5',  compile=False)
aud_model = load_model('audio_model_best.h5',  compile=False)

dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
img_model(dummy)
aud_model(dummy)
print("Models loaded successfully!")
with open('class_indices.json', 'r') as f:
    img_class_indices = json.load(f)
with open('audio_class_indices.json', 'r') as f:
    aud_class_indices = json.load(f)

img_classes = {v: k for k, v in img_class_indices.items()}
aud_classes = {v: k for k, v in aud_class_indices.items()}

shared_species = sorted(set(img_class_indices.keys()) & set(aud_class_indices.keys()))
print(f"Shared species: {len(shared_species)}")

def audio_to_spectrogram_png(audio_path, save_path='temp_spec.png'):
    """Convert wav to mel-spectrogram PNG using the same code as training
    (see audio_utils.py, also used by audio_to_spectrogram.py)."""
    return audio_file_to_spectrogram_png(audio_path, save_path)
def load_spectrogram_for_model(png_path):
    """Load PNG spectrogram and apply preprocess_input — same as training."""
    img = image.load_img(png_path, target_size=(224, 224))
    arr = image.img_to_array(img)          
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)            
    return arr
def get_gradcam(model, input_array):
    inner = None
    for layer in model.layers:
        if hasattr(layer, 'layers'):
            inner = layer
            break

    if inner is not None:
        try:
            grad_model = tf.keras.models.Model(
                inputs=inner.inputs,
                outputs=[inner.get_layer('conv5_block3_out').output,
                         inner.output]
            )
            full_pred  = model(input_array)
            pred_index = tf.argmax(full_pred[0])

            with tf.GradientTape() as tape:
                conv_out, feat = grad_model(input_array)
                x = feat
                found = False
                for layer in model.layers:
                    if layer.name == inner.name:
                        found = True
                        continue
                    if found:
                        try:
                            x = layer(x)
                        except Exception:
                            pass
                loss = x[:, pred_index]
            grads = tape.gradient(loss, conv_out)
        except Exception as e:
            print(f"[Grad-CAM] fallback: {e}")
            grads, conv_out = _gradcam_fallback(model, input_array)
    else:
        grads, conv_out = _gradcam_fallback(model, input_array)
    pooled  = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_out[0] @ pooled[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap).numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap = heatmap / (heatmap.max() + 1e-8)
    return heatmap
def _gradcam_fallback(model, input_array):
    last_conv = None
    for layer in model.layers:
        if 'conv' in layer.name.lower():
            last_conv = layer.name
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(input_array)
        pred_index = tf.argmax(preds[0])
        loss = preds[:, pred_index]
    grads = tape.gradient(loss, conv_out)
    return grads, conv_out
def process_image(img_path):
    img     = image.load_img(img_path, target_size=(224, 224))
    img_arr = image.img_to_array(img)
    preds_list = []
    for aug in [img_arr, np.fliplr(img_arr)]:
        inp = preprocess_input(np.expand_dims(aug.copy(), axis=0))
        preds_list.append(img_model.predict(inp, verbose=0)[0])
    preds = np.mean(preds_list, axis=0)
    inp       = preprocess_input(np.expand_dims(img_arr.copy(), axis=0))
    heatmap   = get_gradcam(img_model, inp)
    heatmap_r = cv2.resize(heatmap, (224, 224))
    heatmap_c = cv2.applyColorMap(np.uint8(255 * heatmap_r), cv2.COLORMAP_JET)
    overlay   = cv2.addWeighted(np.uint8(img_arr), 0.6, heatmap_c, 0.4, 0)

    return img_arr, heatmap_c, overlay, preds
def process_audio(audio_path):
    png_path = audio_to_spectrogram_png(audio_path, 'temp_spec.png')
    inp   = load_spectrogram_for_model(png_path)
    preds = aud_model.predict(inp, verbose=0)[0]
    heatmap   = get_gradcam(aud_model, inp)
    heatmap_r = cv2.resize(heatmap, (224, 224))
    heatmap_c = cv2.applyColorMap(np.uint8(255 * heatmap_r), cv2.COLORMAP_JET)
    mel_display = cv2.imread(png_path)
    mel_display = cv2.cvtColor(mel_display, cv2.COLOR_BGR2RGB)
    overlay     = cv2.addWeighted(mel_display, 0.6, heatmap_c, 0.4, 0)
    if os.path.exists(png_path):
        os.remove(png_path)
    return mel_display, heatmap_c, overlay, preds
def fuse(img_preds, aud_preds):
    img_aligned = np.array([img_preds[img_class_indices[s]] for s in shared_species])
    aud_aligned = np.array([aud_preds[aud_class_indices[s]] for s in shared_species])

    c_img = float(np.max(img_aligned))
    c_aud = float(np.max(aud_aligned))
    w_img = c_img / (c_img + c_aud)
    w_aud = c_aud / (c_img + c_aud)

    fused = w_img * img_aligned + w_aud * aud_aligned
    return fused, w_img, w_aud
def print_top5(preds, class_map, label):
    top5_idx = np.argsort(preds)[::-1][:5]
    medals   = ['1st', '2nd', '3rd', '4th', '5th']
    print(f"\n--- {label} Top-5 ---")
    for i, idx in enumerate(top5_idx):
        print(f"  {medals[i]}: {class_map[idx]:<35} {preds[idx]*100:.2f}%")
print(f"\nProcessing image : {IMAGE_PATH}")
orig, img_heatmap, img_overlay, img_preds = process_image(IMAGE_PATH)

print(f"Processing audio : {AUDIO_PATH}")
mel_disp, aud_heatmap, aud_overlay, aud_preds = process_audio(AUDIO_PATH)

fused_preds, w_img, w_aud = fuse(img_preds, aud_preds)

final_idx     = np.argmax(fused_preds)
final_species = shared_species[final_idx]
fused_conf    = fused_preds[final_idx] * 100
img_conf      = img_preds[np.argmax(img_preds)] * 100
aud_conf      = aud_preds[np.argmax(aud_preds)] * 100

print(f"\n{'='*55}")
print(f"  PREDICTED SPECIES : {final_species}")
print(f"  FUSED CONFIDENCE  : {fused_conf:.2f}%")
print(f"  w_image={w_img:.2f}   w_audio={w_aud:.2f}")
print(f"{'='*55}")

print_top5(img_preds,   img_classes, 'Image Model')
print_top5(aud_preds,   aud_classes, 'Audio Model')

print(f"\n--- Fused Model Top-5 ---")
for i, idx in enumerate(np.argsort(fused_preds)[::-1][:5]):
    print(f"  {['1st','2nd','3rd','4th','5th'][i]}: "
          f"{shared_species[idx]:<35} {fused_preds[idx]*100:.2f}%")
fig = plt.figure(figsize=(24, 11), facecolor='white')
gs  = plt.GridSpec(2, 4, figure=fig, width_ratios=[1, 1, 1, 1.5],
                   left=0.02, right=0.98, top=0.88, bottom=0.06,
                   wspace=0.15, hspace=0.5)
axes = [[fig.add_subplot(gs[r, c]) for c in range(4)] for r in range(2)]
fig.suptitle(
    f'Prediction: {final_species}  |  Fused Conf: {fused_conf:.2f}%  '
    f'|  w_image={w_img:.2f}  w_audio={w_aud:.2f}',
    fontsize=13, fontweight='bold', y=0.97
)

axes[0][0].imshow(orig.astype(np.uint8))
axes[0][0].set_title('Original Image', fontsize=11, pad=8)
axes[0][0].axis('off')

axes[0][1].imshow(cv2.cvtColor(img_heatmap, cv2.COLOR_BGR2RGB))
axes[0][1].set_title('Grad-CAM Heatmap\n(Image)', fontsize=11, pad=8)
axes[0][1].axis('off')

axes[0][2].imshow(cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB))
axes[0][2].set_title('Grad-CAM Overlay\n(Image)', fontsize=11, pad=8)
axes[0][2].axis('off')

top5_img_idx    = np.argsort(img_preds)[::-1][:5]
top5_img_labels = [img_classes[i].replace('-',' ') for i in top5_img_idx]
top5_img_vals   = [img_preds[i] for i in top5_img_idx]
y_pos = np.arange(5)
axes[0][3].barh(y_pos, top5_img_vals[::-1], color='steelblue', height=0.55)
axes[0][3].set_xlim(0, 1.35)
axes[0][3].set_yticks([])
axes[0][3].tick_params(axis='x', labelsize=9)
axes[0][3].set_title(f'Image Model Top-5\n(conf={img_conf:.2f}%)', fontsize=11, pad=8)
for i, (val, name) in enumerate(zip(top5_img_vals[::-1], top5_img_labels[::-1])):
    axes[0][3].text(val + 0.02, i, f'{name}  {val*100:.1f}%',
                    va='center', ha='left', fontsize=9, color='black')
axes[0][3].spines['top'].set_visible(False)
axes[0][3].spines['right'].set_visible(False)
axes[0][3].margins(y=0.2)

axes[1][0].imshow(mel_disp, aspect='auto')
axes[1][0].set_title('Mel-Spectrogram', fontsize=11, pad=8)
axes[1][0].axis('off')

axes[1][1].imshow(aud_heatmap)
axes[1][1].set_title('Grad-CAM Heatmap\n(Spectrogram)', fontsize=11, pad=8)
axes[1][1].axis('off')

axes[1][2].imshow(aud_overlay)
axes[1][2].set_title('Grad-CAM Overlay\n(Spectrogram)', fontsize=11, pad=8)
axes[1][2].axis('off')
top5_aud_idx    = np.argsort(aud_preds)[::-1][:5]
top5_aud_labels = [aud_classes[i].replace('-',' ') for i in top5_aud_idx]
top5_aud_vals   = [aud_preds[i] for i in top5_aud_idx]
axes[1][3].barh(y_pos, top5_aud_vals[::-1], color='seagreen', height=0.55)
axes[1][3].set_xlim(0, 1.35)
axes[1][3].set_yticks([])
axes[1][3].tick_params(axis='x', labelsize=9)
axes[1][3].set_title(f'Audio Model Top-5\n(conf={aud_conf:.2f}%)', fontsize=11, pad=8)
for i, (val, name) in enumerate(zip(top5_aud_vals[::-1], top5_aud_labels[::-1])):
    axes[1][3].text(val + 0.02, i, f'{name}  {val*100:.1f}%',
                    va='center', ha='left', fontsize=9, color='black')
axes[1][3].spines['top'].set_visible(False)
axes[1][3].spines['right'].set_visible(False)
axes[1][3].margins(y=0.2)
plt.savefig('gradcam_fusion_result.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.show()
print("\nDone! Result saved as gradcam_fusion_result.png")