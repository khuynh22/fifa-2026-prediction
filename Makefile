.PHONY: data train evaluate predict test
data:      ; python -m fifa2026.cli data
train:     ; python -m fifa2026.cli train
evaluate:  ; python -m fifa2026.cli evaluate
predict:   ; python -m fifa2026.cli predict
test:      ; pytest -q
