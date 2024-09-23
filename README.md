This is the OpenCLIP fork for our ECCV '24 Beyond Euclidean Workshop paper, [Embedding Geometries of Contrastive Language-Image Pre-Training](https://arxiv.org/abs/2409.13079). The most important new argument is `--geometry`:

https://github.com/EIFY/open_clip/blob/b99af52fcd766a94b3929909b376dfec364de19a/src/training/params.py#L473-L478

where `'euclidean-squared'` corresponds to EuCLIP, `'hyperbolic'` corresponds to MERU, and the default `'clip'` corresponds to the plain CLIP. The entailment loss for Euclidean and hyperbolic geometries can be enabled with `--entailment-weight` and `--min-radius`:

https://github.com/EIFY/open_clip/blob/b99af52fcd766a94b3929909b376dfec364de19a/src/training/params.py#L479-L490

For distance squared logit models (`'euclidean-squared'` and `'hyperbolic-squared'`) you probably need lower initial logit scale to compensate for the quadratic dependence on the distance:

https://github.com/EIFY/open_clip/blob/b99af52fcd766a94b3929909b376dfec364de19a/src/training/params.py#L457-L461

especially for models trained with entailment loss. The distance squared logit models presented in the paper are trained with either `--init-logit-scale 0` (ones without entailment loss) or `--init-logit-scale " -1."` (ones with entailment loss).

Weights for ViT-B/16 and ViT-B/32 EuCLIP / MERU / CLIP models (Sec. 5.1 & 5.2) are [available on Hugging Face](https://huggingface.co/nahidalam/CLIP_Embedding_Geometries).
