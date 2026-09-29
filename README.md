
% GeoShift — LaTeX/TikZ diagrams
% Requires:
% \usepackage{tikz}
% \usetikzlibrary{arrows.meta,positioning,fit,calc,shapes.geometric}

\tikzset{
  geobox/.style={
    rectangle, rounded corners, draw, thick,
    align=center, minimum height=9mm, minimum width=34mm,
    inner sep=4pt
  },
  geoboxwide/.style={
    rectangle, rounded corners, draw, thick,
    align=center, minimum height=9mm, minimum width=48mm,
    inner sep=4pt
  },
  geoboxsmall/.style={
    rectangle, rounded corners, draw, thick,
    align=center, minimum height=8mm, minimum width=25mm,
    inner sep=3pt
  },
  geoarrow/.style={-{Latex[length=2.5mm]}, thick},
  geodash/.style={-{Latex[length=2.5mm]}, thick, dashed},
  geogroup/.style={draw, rounded corners, dashed, inner sep=8pt}
}

% ============================================================
% 1. Overall GeoShift approach
% ============================================================
\begin{figure}[ht]
\centering
\begin{tikzpicture}[node distance=8mm and 13mm]
  \node[geobox] (region) {Region-aware\\validation};
  \node[geobox] (spectral) [right=of region]
    {Multispectral\\features};

  \node[geobox] (sgkf) [below=of region]
    {StratifiedGroupKFold\\by geographic region};
  \node[geobox] (bands) [below=of spectral]
    {B, G, R, NIR\\$+$ NDVI, NDWI};

  \node[geoboxwide] (backbone) [below=13mm of $(sgkf.south)!0.5!(bands.south)$]
    {ConvNeXt-Tiny backbone};

  \node[geoboxwide] (tokens) [below=of backbone]
    {Spatial tokens};

  \node[geoboxwide] (vit) [below=of tokens]
    {Vision Transformer head};

  \node[geoboxwide] (cls) [below=of vit]
    {6-class classification};

  \node[geoboxsmall] (ema) [below left=9mm and -4mm of cls]
    {EMA weights};
  \node[geoboxsmall] (tta) [below=9mm of cls]
    {8-way TTA};
  \node[geoboxsmall] (ensemble) [below right=9mm and -4mm of cls]
    {Fold ensemble};

  \node[geoboxwide] (oof) [below=18mm of $(ema.south)!0.5!(ensemble.south)$]
    {OOF predictions};
  \node[geoboxwide] (bias) [below=of oof]
    {Class-bias tuning};
  \node[geoboxwide] (submission) [below=of bias]
    {Submission};

  \draw[geoarrow] (region) -- (sgkf);
  \draw[geoarrow] (spectral) -- (bands);
  \draw[geoarrow] (sgkf) -- (backbone);
  \draw[geoarrow] (bands) -- (backbone);
  \draw[geoarrow] (backbone) -- (tokens);
  \draw[geoarrow] (tokens) -- (vit);
  \draw[geoarrow] (vit) -- (cls);

  \draw[geoarrow] (cls) -- (ema);
  \draw[geoarrow] (cls) -- (tta);
  \draw[geoarrow] (cls) -- (ensemble);

  \draw[geoarrow] (ema) -- (oof);
  \draw[geoarrow] (tta) -- (oof);
  \draw[geoarrow] (ensemble) -- (oof);

  \draw[geoarrow] (oof) -- (bias);
  \draw[geoarrow] (bias) -- (submission);
\end{tikzpicture}
\caption{Overall GeoShift solution workflow.}
\end{figure}


% ============================================================
% 2. Random 80/20 validation
% ============================================================
\begin{figure}[ht]
\centering
\begin{tikzpicture}[node distance=8mm and 12mm]
  \node[geoboxwide] (data) {Dataset};
  \node[geoboxsmall] (train) [below left=10mm and -4mm of data]
    {Train\\80\%};
  \node[geoboxsmall] (val) [below right=10mm and -4mm of data]
    {Validation\\20\%};

  \draw[geoarrow] (data) -- (train);
  \draw[geoarrow] (data) -- (val);
\end{tikzpicture}
\caption{Initial random 80/20 validation strategy.}
\end{figure}


% ============================================================
% 3. Region-aware validation
% ============================================================
\[
\boxed{
\mathrm{Validation\ regions}\cap
\mathrm{Training\ regions}
=
\varnothing
}
\]

\begin{figure}[ht]
\centering
\begin{tikzpicture}[node distance=8mm]
  \node[geoboxwide] (data) {All labelled samples};
  \node[geoboxwide] (sgkf) [below=of data]
    {StratifiedGroupKFold};
  \node[geoboxsmall] (train) [below left=10mm and -3mm of sgkf]
    {Training\\regions};
  \node[geoboxsmall] (val) [below right=10mm and -3mm of sgkf]
    {Validation\\regions};

  \draw[geoarrow] (data) -- (sgkf);
  \draw[geoarrow] (sgkf) -- (train);
  \draw[geoarrow] (sgkf) -- (val);
\end{tikzpicture}
\caption{Region-aware validation with geographic groups.}
\end{figure}


% ============================================================
% 4. Old vs improved spatial tokenisation
% ============================================================
\begin{figure}[ht]
\centering
\begin{tikzpicture}[node distance=9mm and 18mm]
  \node[geobox] (oldmap) {$2\times2$\\feature map};
  \node[geobox] (oldtok) [right=of oldmap] {4\\tokens};

  \node[geobox] (newmap) [below=14mm of oldmap]
    {$4\times4$\\feature map};
  \node[geobox] (newtok) [right=of newmap] {16\\tokens};

  \draw[geoarrow] (oldmap) -- (oldtok);
  \draw[geoarrow] (newmap) -- (newtok);

  \node[above=2mm of oldmap] {\textbf{Earlier design}};
  \node[above=2mm of newmap] {\textbf{Final default}};
\end{tikzpicture}
\caption{Increase in the number of spatial tokens presented to the Transformer.}
\end{figure}


% ============================================================
% 5. High-resolution preset
% ============================================================
\begin{figure}[ht]
\centering
\begin{tikzpicture}[node distance=10mm]
  \node[geobox] (map) {$8\times8$\\feature map};
  \node[geobox] (tokens) [below=of map]
    {64 spatial\\tokens};
  \draw[geoarrow] (map) -- (tokens);
\end{tikzpicture}
\caption{Higher-resolution feature-map option (\texttt{stem\_stride=2}).}
\end{figure}


% ============================================================
% 6. Multispectral feature construction
% ============================================================
\begin{figure}[ht]
\centering
\begin{tikzpicture}[node distance=7mm and 11mm]
  \node[geoboxsmall] (b) {Blue};
  \node[geoboxsmall] (g) [right=of b] {Green};
  \node[geoboxsmall] (r) [right=of g] {Red};
  \node[geoboxsmall] (nir) [right=of r] {NIR};

  \node[geoboxsmall] (ndvi) [below=10mm of $(b.south)!0.5!(g.south)$]
    {NDVI};
  \node[geoboxsmall] (ndwi) [right=of ndvi]
    {NDWI};

  \node[geoboxwide] (input) [below=11mm of $(ndvi.south)!0.5!(ndwi.south)$]
    {6-channel model input\\B, G, R, NIR, NDVI, NDWI};

  \draw[geoarrow] (b) -- (input);
  \draw[geoarrow] (g) -- (input);
  \draw[geoarrow] (r) -- (input);
  \draw[geoarrow] (nir) -- (input);
  \draw[geoarrow] (ndvi) -- (input);
  \draw[geoarrow] (ndwi) -- (input);
\end{tikzpicture}
\caption{Construction of the six-channel multispectral input.}
\end{figure}


% ============================================================
% 7. Final model architecture
% ============================================================
\begin{figure}[ht]
\centering
\begin{tikzpicture}[node distance=8mm]
  \node[geoboxwide] (input)
    {Input: $64\times64\times6$\\B, G, R, NIR, NDVI, NDWI};

  \node[geoboxwide] (stem) [below=of input]
    {Modified ConvNeXt-Tiny stem};

  \node[geoboxwide] (stages) [below=of stem]
    {ConvNeXt stages 1--3};

  \node[geoboxwide] (map) [below=of stages]
    {$4\times4\times384$ feature map};

  \node[geoboxwide] (tokens) [below=of map]
    {16 spatial tokens};

  \node[geoboxwide] (norm) [below=of tokens]
    {LayerNorm + linear token projection};

  \node[geoboxwide] (cls) [below=of norm]
    {[CLS] token + positional embeddings};

  \node[geoboxwide] (transformer) [below=of cls]
    {2-layer Transformer encoder};

  \node[geoboxwide] (headnorm) [below=of transformer]
    {LayerNorm};

  \node[geoboxwide] (mlp) [below=of headnorm]
    {MLP classification head};

  \node[geoboxwide] (out) [below=of mlp]
    {6 output logits};

  \draw[geoarrow] (input) -- (stem);
  \draw[geoarrow] (stem) -- (stages);
  \draw[geoarrow] (stages) -- (map);
  \draw[geoarrow] (map) -- (tokens);
  \draw[geoarrow] (tokens) -- (norm);
  \draw[geoarrow] (norm) -- (cls);
  \draw[geoarrow] (cls) -- (transformer);
  \draw[geoarrow] (transformer) -- (headnorm);
  \draw[geoarrow] (headnorm) -- (mlp);
  \draw[geoarrow] (mlp) -- (out);
\end{tikzpicture}
\caption{Final ConvNeXt-Tiny + ViT architecture.}
\end{figure}


% ============================================================
% 8. Pretrained weight adaptation
% ============================================================
\begin{figure}[ht]
\centering
\begin{tikzpicture}[node distance=8mm and 13mm]
  \node[geoboxwide] (pre)
    {Pretrained RGB convolution};

  \node[geoboxwide] (stem) [below=of pre]
    {Modified multispectral stem};

  \node[geoboxsmall] (rgb) [below left=11mm and -2mm of stem]
    {RGB\\channels};
  \node[geoboxsmall] (nir) [below=11mm of stem]
    {NIR\\channel};
  \node[geoboxsmall] (idx) [below right=11mm and -2mm of stem]
    {NDVI/NDWI\\channels};

  \draw[geoarrow] (pre) -- (stem);
  \draw[geoarrow] (stem) -- (rgb);
  \draw[geoarrow] (stem) -- (nir);
  \draw[geoarrow] (stem) -- (idx);

  \node[align=left, below=7mm of rgb, xshift=15mm]
    {\small RGB $\rightarrow$ corresponding pretrained filters\\
     NIR $\rightarrow$ mean pretrained filter\\
     NDVI/NDWI $\rightarrow$ small initial weights};
\end{tikzpicture}
\caption{Adaptation of pretrained RGB weights to multispectral input.}
\end{figure}


% ============================================================
% 9. Per-fold training methodology
% ============================================================
\begin{figure}[ht]
\centering
\begin{tikzpicture}[node distance=7mm]
  \node[geobox] (split) {1. Split\\by region};
  \node[geobox] (stats) [below=of split]
    {2. Train-only\\normalisation};
  \node[geobox] (model) [below=of stats]
    {3. Build\\model};
  \node[geobox] (optim) [below=of model]
    {4. Initialise\\optimizer};
  \node[geobox] (train) [below=of optim]
    {5. Train with\\augmentation};
  \node[geobox] (ema) [below=of train]
    {6. Update\\EMA};
  \node[geobox] (eval) [below=of ema]
    {7. Validate\\Macro-F1};
  \node[geobox] (ckpt) [below=of eval]
    {8. Save best\\EMA checkpoint};
  \node[geobox] (oof) [below=of ckpt]
    {9. Generate\\OOF predictions};
  \node[geobox] (test) [below=of oof]
    {10. Generate\\test probabilities};

  \foreach \a/\b in {
    split/stats,
    stats/model,
    model/optim,
    optim/train,
    train/ema,
    ema/eval,
    eval/ckpt,
    ckpt/oof,
    oof/test}
    \draw[geoarrow] (\a) -- (\b);
\end{tikzpicture}
\caption{Operations performed for each cross-validation fold.}
\end{figure}


% ============================================================
% 10. Learning-rate strategy
% ============================================================
\[
\boxed{
\mathrm{LR}(t)=
\mathrm{base\_LR}\times
\mathrm{cosine\_factor}(t)\times
\mathrm{plateau\_scale}
}
\]

\begin{figure}[ht]
\centering
\begin{tikzpicture}[node distance=8mm and 14mm]
  \node[geoboxsmall] (base) {Base LR};
  \node[geoboxsmall] (cos) [right=of base] {Warm-up +\\cosine factor};
  \node[geoboxsmall] (plateau) [right=of cos] {Persistent\\plateau scale};
  \node[geoboxwide] (lr) [below=12mm of cos]
    {Final learning rate};

  \draw[geoarrow] (base) -- (cos);
  \draw[geoarrow] (cos) -- (lr);
  \draw[geoarrow] (plateau) -- (lr);
\end{tikzpicture}
\caption{Final learning-rate control after fixing the plateau-scheduler interaction.}
\end{figure}


% ============================================================
% 11. Five-fold region-aware CV
% ============================================================
\begin{figure}[ht]
\centering
\begin{tikzpicture}[node distance=8mm and 12mm]
  \node[geoboxwide] (data) {All samples};
  \node[geoboxwide] (cv) [below=of data]
    {5-fold StratifiedGroupKFold\\group = geographic region};

  \node[geoboxsmall] (f0) [below left=12mm and 20mm of cv]
    {Fold 0};
  \node[geoboxsmall] (f1) [right=of f0] {Fold 1};
  \node[geoboxsmall] (f2) [right=of f1] {Fold 2};
  \node[geoboxsmall] (f3) [right=of f2] {Fold 3};
  \node[geoboxsmall] (f4) [right=of f3] {Fold 4};

  \draw[geoarrow] (data) -- (cv);
  \foreach \f in {f0,f1,f2,f3,f4}
    \draw[geoarrow] (cv) -- (\f);
\end{tikzpicture}
\caption{Five-fold region-aware cross-validation.}
\end{figure}


% ============================================================
% 12. Example region split
% ============================================================
\begin{figure}[ht]
\centering
\begin{tikzpicture}[node distance=10mm and 14mm]
  \node[geoboxwide] (fold) {Fold 0};

  \node[geoboxwide] (train) [below left=12mm and -2mm of fold]
    {Training regions\\R01, R02, R03, R04, \ldots, R20};

  \node[geoboxwide] (val) [below right=12mm and -2mm of fold]
    {Validation regions\\R21, R22, R23};

  \draw[geoarrow] (fold) -- (train);
  \draw[geoarrow] (fold) -- (val);
\end{tikzpicture}
\caption{Illustrative region separation for one fold.}
\end{figure}


% ============================================================
% 13. OOF reconstruction and uses
% ============================================================
\begin{figure}[ht]
\centering
\begin{tikzpicture}[node distance=8mm and 13mm]
  \node[geoboxsmall] (oof0) {OOF\\Fold 0};
  \node[geoboxsmall] (oof1) [right=of oof0] {OOF\\Fold 1};
  \node[geoboxsmall] (dots) [right=of oof1] {$\cdots$};
  \node[geoboxsmall] (oof4) [right=of dots] {OOF\\Fold 4};

  \node[geoboxwide] (all) [below=14mm of $(oof0.south)!0.5!(oof4.south)$]
    {Complete OOF prediction set};

  \node[geoboxsmall] (cm) [below left=12mm and 3mm of all]
    {Confusion\\matrix};
  \node[geoboxsmall] (cls) [below=12mm of all]
    {Per-class\\analysis};
  \node[geoboxsmall] (reg) [below right=12mm and 3mm of all]
    {Region-wise\\errors};
  \node[geoboxsmall] (bias) [below=12mm of cls]
    {Class-bias\\tuning};

  \draw[geoarrow] (oof0) -- (all);
  \draw[geoarrow] (oof1) -- (all);
  \draw[geoarrow] (dots) -- (all);
  \draw[geoarrow] (oof4) -- (all);

  \draw[geoarrow] (all) -- (cm);
  \draw[geoarrow] (all) -- (cls);
  \draw[geoarrow] (all) -- (reg);
  \draw[geoarrow] (all) -- (bias);
\end{tikzpicture}
\caption{How fold-level OOF predictions are combined and analysed.}
\end{figure}


% ============================================================
% 14. Eight-way TTA
% ============================================================
\begin{figure}[ht]
\centering
\begin{tikzpicture}[node distance=7mm and 9mm]
  \node[geoboxsmall] (a) {Original};
  \node[geoboxsmall] (b) [right=of a] {Rotate 90$^\circ$};
  \node[geoboxsmall] (c) [right=of b] {Rotate 180$^\circ$};
  \node[geoboxsmall] (d) [right=of c] {Rotate 270$^\circ$};

  \node[geoboxsmall] (e) [below=10mm of a] {Flip};
  \node[geoboxsmall] (f) [right=of e] {Flip + 90$^\circ$};
  \node[geoboxsmall] (g) [right=of f] {Flip + 180$^\circ$};
  \node[geoboxsmall] (h) [right=of g] {Flip + 270$^\circ$};

  \node[geoboxwide] (avg) [below=13mm of $(e.south)!0.5!(h.south)$]
    {$P_{\mathrm{TTA}}=\frac{1}{8}\sum_{i=1}^{8}P_i$};

  \foreach \n in {a,b,c,d,e,f,g,h}
    \draw[geoarrow] (\n) -- (avg);
\end{tikzpicture}
\caption{Eight-way dihedral test-time augmentation and probability averaging.}
\end{figure}


% ============================================================
% 15. Fold ensemble
% ============================================================
\[
\boxed{
P_{\mathrm{ensemble}}
=
\frac{1}{K}
\sum_{k=1}^{K} P_{\mathrm{fold}_k}
}
\]

\begin{figure}[ht]
\centering
\begin{tikzpicture}[node distance=8mm and 14mm]
  \node[geoboxsmall] (f0) {Fold 0\\probabilities};
  \node[geoboxsmall] (f1) [right=of f0] {Fold 1\\probabilities};
  \node[geoboxsmall] (dots) [right=of f1] {$\cdots$};
  \node[geoboxsmall] (f4) [right=of dots] {Fold 4\\probabilities};

  \node[geoboxwide] (ens) [below=13mm of $(f0.south)!0.5!(f4.south)$]
    {Mean probability ensemble};

  \draw[geoarrow] (f0) -- (ens);
  \draw[geoarrow] (f1) -- (ens);
  \draw[geoarrow] (dots) -- (ens);
  \draw[geoarrow] (f4) -- (ens);
\end{tikzpicture}
\caption{Probability-level fold ensembling.}
\end{figure}


% ============================================================
% 16. OOF class-bias tuning
% ============================================================
\begin{figure}[ht]
\centering
\begin{tikzpicture}[node distance=9mm and 13mm]
  \node[geoboxwide] (oof) {OOF probabilities};
  \node[geoboxwide] (logp) [below=of oof]
    {$\log(P_{\mathrm{class}})$};
  \node[geoboxwide] (bias) [below=of logp]
    {$\log(P_{\mathrm{class}})+b_{\mathrm{class}}$};
  \node[geoboxwide] (search) [below=of bias]
    {Search class offsets for higher OOF Macro-F1};
  \node[geoboxwide] (sub) [below=of search]
    {Submission with tuned class offsets};

  \draw[geoarrow] (oof) -- (logp);
  \draw[geoarrow] (logp) -- (bias);
  \draw[geoarrow] (bias) -- (search);
  \draw[geoarrow] (search) -- (sub);
\end{tikzpicture}
\caption{OOF-based class log-probability bias tuning.}
\end{figure}


% ============================================================
% 17. Complete final pipeline
% ============================================================
\begin{figure}[ht]
\centering
\begin{tikzpicture}[node distance=6.5mm]
  \node[geoboxwide] (raw) {Raw data};
  \node[geoboxwide] (region) [below=of raw] {Region extraction};
  \node[geoboxwide] (cv) [below=of region] {StratifiedGroupKFold};

  \node[geoboxsmall] (fold0) [below left=11mm and 7mm of cv]
    {Fold 0};
  \node[geoboxsmall] (dots) [right=of fold0] {$\cdots$};
  \node[geoboxsmall] (fold4) [right=of dots] {Fold 4};

  \node[geoboxwide] (trainstats) [below=12mm of $(fold0.south)!0.5!(fold4.south)$]
    {Train-only statistics};

  \node[geoboxwide] (aug) [below=of trainstats]
    {Data augmentation};

  \node[geoboxwide] (model) [below=of aug]
    {ConvNeXt + ViT};

  \node[geoboxwide] (ema) [below=of model]
    {EMA model + best checkpoint};

  \node[geoboxwide] (oof) [below=of ema]
    {OOF predictions};

  \node[geoboxwide] (analysis) [below=of oof]
    {Error analysis};

  \node[geoboxwide] (bias) [below=of analysis]
    {Class-bias tuning};

  \node[geoboxwide] (test) [below=of bias]
    {Test predictions};

  \node[geoboxwide] (tta) [below=of test]
    {8-way TTA};

  \node[geoboxwide] (ensemble) [below=of tta]
    {Fold ensemble};

  \node[geoboxwide] (submission) [below=of ensemble]
    {Submission};

  \foreach \a/\b in {
    raw/region,
    region/cv,
    trainstats/aug,
    aug/model,
    model/ema,
    ema/oof,
    oof/analysis,
    analysis/bias,
    bias/test,
    test/tta,
    tta/ensemble,
    ensemble/submission}
    \draw[geoarrow] (\a) -- (\b);

  \draw[geoarrow] (cv) -- (fold0);
  \draw[geoarrow] (cv) -- (dots);
  \draw[geoarrow] (cv) -- (fold4);
  \draw[geoarrow] (fold0) -- (trainstats);
  \draw[geoarrow] (dots) -- (trainstats);
  \draw[geoarrow] (fold4) -- (trainstats);
\end{tikzpicture}
\caption{End-to-end GeoShift training and inference pipeline.}
\end{figure}


% ============================================================
% 18. Failure diagnosis tree
% ============================================================
\begin{figure}[ht]
\centering
\begin{tikzpicture}[node distance=8mm and 10mm]
  \node[geoboxwide] (wrong) {Wrong prediction};

  \node[geoboxsmall] (spectral) [below left=13mm and 20mm of wrong]
    {Spectral\\ambiguity?};
  \node[geoboxsmall] (spatial) [right=of spectral]
    {Spatial\\ambiguity?};
  \node[geoboxsmall] (imbalance) [right=of spatial]
    {Class\\imbalance?};
  \node[geoboxsmall] (region) [right=of imbalance]
    {Region-specific\\appearance?};

  \node[geoboxsmall] (resolution) [below=12mm of spatial]
    {Insufficient\\resolution?};
  \node[geoboxsmall] (label) [right=of resolution]
    {Annotation\\ambiguity?};

  \draw[geoarrow] (wrong) -- (spectral);
  \draw[geoarrow] (wrong) -- (spatial);
  \draw[geoarrow] (wrong) -- (imbalance);
  \draw[geoarrow] (wrong) -- (region);
  \draw[geoarrow] (wrong) -- (resolution);
  \draw[geoarrow] (wrong) -- (label);
\end{tikzpicture}
\caption{Failure-analysis framework for misclassified samples.}
\end{figure}


% ============================================================
% 19. Final development loop
% ============================================================
\begin{figure}[ht]
\centering
\begin{tikzpicture}[node distance=8mm and 13mm]
  \node[geoboxwide] (initial) {Initial model};
  \node[geoboxwide] (failure) [below=of initial]
    {Observed failure};
  \node[geoboxwide] (cause) [below=of failure]
    {Identify cause};
  \node[geoboxwide] (change) [below=of cause]
    {Change methodology / architecture};
  \node[geoboxwide] (evaluate) [below=of change]
    {Evaluate with region-aware validation};
  \node[geoboxwide] (repeat) [below=of evaluate]
    {Repeat};
  \node[geoboxwide] (final) [below=of repeat]
    {Final ensemble pipeline};

  \foreach \a/\b in {
    initial/failure,
    failure/cause,
    cause/change,
    change/evaluate,
    evaluate/repeat,
    repeat/final}
    \draw[geoarrow] (\a) -- (\b);

  \draw[geodash] (repeat.west) .. controls +(-2,0) and +(-2,0)
    .. (failure.west);
\end{tikzpicture}
\caption{Iterative model-development process.}
\end{figure}
