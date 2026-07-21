$ErrorActionPreference = 'Stop'

$src = 'doc/MasterThesis/chapters'
$dst = 'doc/BDRPReportTemplate'

function Body($name) {
    $p = Join-Path $src $name
    $t = Get-Content $p -Raw
    $t = $t -replace '(?m)^\\chapter\*?\{[^}]+\}\r?\n', ''
    $t = $t -replace '(?m)^\\addcontentsline\{toc\}\{chapter\}\{[^}]+\}\r?\n', ''
    return $t.Trim()
}

$intro = Body 'introduction.tex'
$phys = Body 'ch1_physical_framework.tex'
$dim = Body 'ch2_dim_reduction.tex'
$diff = Body 'ch3_diffusion_model.tex'
$metrics = Body 'ch4_metrics.tex'
$wave = Body 'ch5_wavelet_diffusion.tex'
$concl = Body 'conclusion.tex'
$ack = Body 'acknowledgements.tex'
$abs = Body 'abstract.tex'

@"
\chapter{Introduction}
\label{chap:introduction}

$intro

\section{Project Objectives and Contributions}

The internship objective is to design and evaluate generative surrogate models for two-dimensional Rayleigh--Taylor instability fields. The main contributions of this work are:
\begin{itemize}
\item the implementation of a baseline diffusion model trained directly on preprocessed density fields;
\item the construction of physically consistent data augmentation strategies adapted to the symmetries and periodicity of the dataset;
\item the study of Fourier and wavelet representations as tools for dimensionality reduction and multi-scale analysis;
\item the design of a wavelet-based diffusion pipeline intended to exploit the scale-separated structure of RTI fields;
\item the definition and use of evaluation metrics combining feature-space distances and physical statistical indicators.
\end{itemize}

\section{Report Outline}

Chapter~\ref{chap:related-work} positions the project with respect to existing scientific and machine-learning approaches. Chapter~\ref{chap:background} introduces the physical, numerical, and mathematical background required to understand the work. Chapter~\ref{chap:methodology} details the implemented methodology. Chapter~\ref{chap:experiments} presents the dataset, experiments, metrics, and results. Chapter~\ref{chap:conclusion} summarizes the work and discusses limitations and perspectives.
"@ | Set-Content (Join-Path $dst 'Chapter1.tex') -Encoding UTF8

@"
\chapter{Related Work}
\label{chap:related-work}

This chapter positions the project with respect to the main scientific and methodological families used in the report: high-fidelity numerical simulation of Rayleigh--Taylor instability, spectral or multi-resolution representations of physical fields, and generative modeling for surrogate simulation. It should be completed with precise bibliographic references before final submission.

\section{High-Fidelity Simulation of Rayleigh--Taylor Instability}

Direct Numerical Simulations are used as reference data because they resolve the relevant spatial and temporal scales of the flow without relying on turbulence closure models. In the context of RTI, they make it possible to observe the transition from early interface perturbations to non-linear bubble, spike, and mixing structures. Their main limitation is computational cost, which restricts the number of available realizations and motivates data-driven surrogate modeling.

\textbf{TODO:} Add the main RTI/DNS references used by the CEA team, including the numerical solver, physical assumptions, and any previous studies on mono-mode or bi-mode RTI configurations.

\section{Spectral and Multi-resolution Representations}

Fourier methods are natural for periodic flows and turbulence analysis because they describe a field in terms of global frequency modes. They are particularly useful for energy-spectrum analysis and pseudo-spectral solvers. However, global basis functions are less adapted to localized structures and sharp gradients.

Wavelet methods provide a complementary representation localized in both space and scale. This property is useful for RTI fields, where large coherent structures coexist with localized small-scale details. The multi-resolution structure also makes wavelets attractive for compression, denoising, and scale-aware learning.

\textbf{TODO:} Add references on Fourier analysis in turbulence, wavelet compression or JPEG2000, and wavelet-based analysis of turbulent or multi-scale physical fields.

\section{Generative Surrogate Models for Physical Fields}

Generative models aim to learn a probability distribution from reference samples and then generate new realizations from this distribution. Diffusion models are particularly relevant because they can represent complex high-dimensional distributions and have shown strong empirical performance in image generation. For physical surrogate modeling, the challenge is not only visual realism but also the preservation of statistical and physical properties.

\textbf{TODO:} Add related work on diffusion models, score-based generative modeling, DDPMs, and generative modeling for scientific simulation or turbulent fields. Explain how the present work differs from these approaches through the combination of RTI data, physically motivated augmentation, PhyFID-style evaluation, and wavelet conditioning.

\section{Motivation for the Chosen Direction}

The reviewed approaches suggest that a direct diffusion baseline is a necessary reference point, while wavelet representations may introduce a useful inductive bias for limited and multi-scale physical datasets. This motivates the two-pipeline strategy developed in the following chapters.
"@ | Set-Content (Join-Path $dst 'Chapter2.tex') -Encoding UTF8

@"
\chapter{Background}
\label{chap:background}

This chapter introduces the concepts needed to understand the proposed methodology: the physical and numerical framework of Rayleigh--Taylor instability, dimensionality reduction through Fourier and wavelet representations, and the principles of score-based diffusion models.

\section{Physical and Numerical Framework}

$phys

\section{Dimensionality Reduction and Multi-scale Representations}

$dim

\section{Diffusion Models for RTI Fields}

$diff

This background chapter introduced the physical, numerical, and algorithmic foundations of the work. The next chapter builds on these elements to describe the implemented methodology and the two generative pipelines studied during the internship.
"@ | Set-Content (Join-Path $dst 'Chapter3.tex') -Encoding UTF8

@"
\chapter{Our Methodology and Approach}
\label{chap:methodology}

This chapter describes the methodology followed during the internship. The approach is organized around two generative pipelines: a baseline diffusion model in physical image space and a wavelet-based diffusion model using multi-resolution coefficients.

\section{Overall Pipeline}

The complete workflow starts from Direct Numerical Simulation density fields, applies preprocessing and physically motivated data augmentation, trains a generative diffusion model, and evaluates generated samples against validation data. Figure~\ref{fig:diffusion_pipeline} gives the general principle of this surrogate modeling approach.

\textbf{TODO:} Add or update a dedicated methodology figure showing all components expected by the BDRP template: raw DNS data, preprocessing, augmentation, baseline diffusion, wavelet decomposition, wavelet diffusion, reconstruction, and evaluation.

\section{Baseline Image-space Diffusion Pipeline}

The baseline model is trained directly on preprocessed two-dimensional RTI density fields. This pipeline is used as the reference against which the wavelet-based approach is compared.

\textbf{TODO:} Summarize implementation choices in pseudo-code: preprocessing, training loop, noise sampling, loss computation, validation sampling, and saving generated datasets. Avoid source code in the report body.

\section{Wavelet-based Diffusion Pipeline}

$wave

\section{Implementation Contributions}

\textbf{TODO:} Complete this section with the concrete software contributions made during the internship: repository organization, training scripts, dataset loaders, augmentation code, evaluation scripts, configuration files, reproducibility instructions, and computational environment.

\section{Reproducibility}

The GitHub link must appear on the title page and the repository should include a README explaining how to run the program and how to access or prepare the dataset.

\textbf{TODO:} Add the final GitHub URL on the title page and verify supervisor access. Add or update the repository README with execution commands and dataset instructions.

This methodology chapter detailed how the baseline and wavelet-based models are constructed. The next chapter evaluates these choices experimentally using visual, statistical, and physically motivated indicators.
"@ | Set-Content (Join-Path $dst 'Chapter4.tex') -Encoding UTF8

@"
\chapter{Experiments and Evaluation}
\label{chap:experiments}

This chapter presents the experimental protocol used to evaluate the generated RTI fields. The objective is to compare generated samples with validation DNS fields using both generic feature-space metrics and physical indicators adapted to the Rayleigh--Taylor problem.

\section{Dataset Description}

The dataset consists of two-dimensional density fields extracted from Direct Numerical Simulations of Rayleigh--Taylor instability. The fields contain coherent large-scale structures and smaller-scale mixing patterns, while the number of available realizations remains limited because of the computational cost of DNS.

\textbf{TODO:} Add exact dataset details: number of simulations, number of snapshots, spatial resolution, train/validation/test split, physical parameters, mono-mode or bi-mode configuration, and storage format.

\section{Experimental Objectives}

The experiments compare the direct image-space diffusion model with the wavelet-based diffusion model. The evaluation focuses on visual realism, convergence behavior, statistical similarity to validation data, preservation of vertical density profiles, and reproduction of scale-dependent fluctuation spectra.

\section{Evaluation Metrics}

$metrics

\section{Baseline Results}

The baseline diffusion model generates samples that reproduce the main organization of the reference simulations and remain significantly closer to the validation dataset than pure noise or unrelated images. In PhyFID space, the generated fields obtain a distance of approximately $0.132$ from the validation set, compared with $6.04$ for pure noise.

\textbf{TODO:} Add final tables and figures for all reported metrics. Each figure and table must be explicitly cited and interpreted in the text.

\section{Wavelet-based Results}

The wavelet-based formulation introduces a multi-resolution inductive bias and shows that diffusion can learn relationships between large-scale approximations and fine-scale details. In its current implementation, reconstructed fields remain rougher than those obtained with the direct baseline, which indicates that the conditioning mechanism and architecture still need refinement.

\textbf{TODO:} Evaluate the wavelet-based model with the same metrics as the baseline: PhyFID, FID, line-wise statistics, and fluctuation spectra. Add qualitative comparisons of reference, baseline, wavelet reconstruction, and generated fields.

\section{Interpretation}

The current results show that direct diffusion is the strongest baseline in terms of generation quality, while wavelet diffusion provides a more interpretable and physically meaningful direction for future improvement. The comparison highlights the importance of adapting architectures and normalization strategies to transformed coefficient spaces.

This evaluation chapter showed the strengths and limitations of the implemented pipelines. The final chapter summarizes the main findings and discusses perspectives for improving the wavelet-based approach.
"@ | Set-Content (Join-Path $dst 'Chapter5.tex') -Encoding UTF8

@"
\chapter{Conclusion and Perspectives}
\label{chap:conclusion}

$concl

\section{Limitations}

The main limitations of this work are the small size of the DNS dataset, the partial quantitative evaluation of the wavelet-based model, and the current roughness of reconstructed wavelet-generated fields. The diffusion architecture and conditioning strategy were initially designed close to image-space generation and may not be optimal for the statistical structure of wavelet coefficients.

\section{Perspectives}

Future work should evaluate the wavelet model with the same full metric suite as the baseline, investigate alternative wavelet families and decomposition levels, improve coefficient normalization, and explore progressive coarse-to-fine generation. Larger and more diverse DNS datasets would also make it possible to assess generalization more robustly.

\section{Teamwork and Internship Environment}

\textbf{TODO:} Complete this final paragraph with the work in team, as requested by the BDRP template: supervision at CEA, interactions with PhD students and laboratory members, collaboration practices, meetings, code review or scientific review process, and how these exchanges influenced the project.
"@ | Set-Content (Join-Path $dst 'Chapter6.tex') -Encoding UTF8

@"
\chapter{Additional Material}

\textbf{TODO:} Add appendices if needed: extended experimental tables, additional qualitative samples, implementation details, hyperparameters, dataset preprocessing details, or pseudo-code that is too long for the main report.
"@ | Set-Content (Join-Path $dst 'Appendix1.tex') -Encoding UTF8

$ack | Set-Content (Join-Path $dst 'Acknowledgments.tex') -Encoding UTF8
$abs | Set-Content (Join-Path $dst 'AbstractContent.tex') -Encoding UTF8
