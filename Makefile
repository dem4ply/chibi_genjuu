ignore_test = test/**,venv/*,setup.py

test:: test_unit report

modules = chibi_genjuu

style_test: flakes pep8

test_unit:
	@echo "Running tests"
	@coverage run -m unittest discover -p "*.py" -s tests

report:
	@coverage report
	@coverage html

open_report_firefox:
	@nohup firefox .coverage_html_report/index.html > /dev/null &

clean:
	@echo "Running clean..."
	@find . -name ".*.sw*" -exec rm {} +
	@rm -rf .coverage .coverage_html_report

pep8:
	@echo "Running pep8 tests..."
	@pep8 --statistics ${modules}

flakes:
	@echo "Running flakes tests..."
	@pyflakes ${modules}
