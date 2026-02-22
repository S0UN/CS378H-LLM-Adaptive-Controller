.PHONY: help test test-integration

help:
	@echo "Available targets:"
	@echo "  make test   - run unit tests"
	@echo "  make test-integration   - run live API integration tests (requires OPENAI_API_KEY and RUN_INTEGRATION=1)"

test:
	python3 -m unittest discover -s tests -p 'test_*.py' -v

test-integration:
	RUN_INTEGRATION=1 python3 -m unittest discover -s tests -p 'test_integration_*.py' -v
