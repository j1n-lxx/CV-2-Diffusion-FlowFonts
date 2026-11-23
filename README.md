# CV-2-Diffusion-FlowFonts
FlowFont generates Korean handwriting glyph by glyph in real time. A rectified-flow U-Net conditioned on jamo and writer style is pretrained on open fonts, fine-tuned on Nanum Handwriting, then reflowed and distilled to one network evaluation. On held-out syllables it keeps 97.6% jamo accuracy with 128× fewer evaluations than 64-step diffusion.
