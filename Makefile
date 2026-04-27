.PHONY: all clean

all: presentation.pdf

presentation.pdf: presentation.tex
	lualatex -interaction=nonstopmode presentation.tex
	lualatex -interaction=nonstopmode presentation.tex

clean:
	rm -f presentation.{aux,log,nav,out,snm,toc,vrb}
