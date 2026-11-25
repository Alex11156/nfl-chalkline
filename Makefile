# Chalkline: every routine command in one place. `make` lists them.
# Uses the project's conda env when it exists; override with `make PY=python ...`.
PY ?= $(shell [ -x /opt/anaconda3/envs/nfl/bin/python ] && echo /opt/anaconda3/envs/nfl/bin/python || echo python)

.PHONY: help refresh publish models site props grade app test report guide importance
.DEFAULT_GOAL := help

help:      ## list these commands
	@grep -E '^[a-z]+: .*##' $(MAKEFILE_LIST) | awk -F': .*## ' '{printf "  make %-11s %s\n", $$1, $$2}'

refresh:   ## Tuesday: new data -> features -> models -> grades -> site (kept local)
	$(PY) scripts/refresh_data.py

publish:   ## the same, then commit and push the site
	$(PY) scripts/refresh_data.py --publish

models:    ## backtest and retrain the game and player models on the data already downloaded
	$(PY) -m nfl_engine.models.game_models
	$(PY) -m nfl_engine.models.player_models

site:      ## rebuild the website from the current models
	$(PY) scripts/build_site.py

props:     ## Saturday: save sportsbook prop lines next to the model's odds (spends API credits)
	$(PY) scripts/props.py snapshot

grade:     ## settle every saved prop line against what happened
	$(PY) scripts/props.py grade

app:       ## the local question-answering app
	$(PY) -m streamlit run app/app.py

test:      ## 40 plain-English questions + train/serve parity for the last finished week
	$(PY) tests/test_questions.py
	$(PY) tests/test_parity.py

importance: ## which inputs matter most (reports/research/feature_importance.json)
	$(PY) research/feature_importance.py

report:    ## rebuild reports/Chalkline_Model_Report.pdf: which model does what, how accurate, and why
	$(PY) scripts/report_model.py

guide:     ## rebuild reports/Chalkline_Codebase_Guide.pdf from the saved results
	$(PY) scripts/report_codebase_guide.py
