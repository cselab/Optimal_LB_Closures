.POSIX:
.SUFFIXES:
LATEXMK = latexmk
main.pdf: main.tex refs.bib
	$(LATEXMK) -pdf $< || rm $@

mostlyclean:
	$(LATEXMK) -silent -c -f
	rm *.bbl *-blx.bib *.fff

clean: mostlyclean
	$(LATEXMK) -silent -C -f
	-rm *.run.xml *.synctex.gz
	-rm *.d

main.pdf: \
algorithmic.sty\
algorithm.sty\
fancyhdr.sty\
figures/advection/actions/action_cluster.pdf\
figures/advection/actions/action_imgs.jpg\
figures/advection/low_MAE_0.01_violin400steps_100sims.pdf\
figures/advection/outperformance_evol_400steps_100sims.pdf\
figures/advection/relative_MAE_evol_100steps_100sims.pdf\
figures/advection/relative_MAE_evol_400steps_100sims.pdf\
figures/advection/training/train_length.png\
figures/advection/training/train_reward.png\
figures/burgers/improvement_rel_MAE_evol_200steps.pdf\
figures/burgers/low_MAE_0.1_violin400steps_100sims.pdf\
figures/burgers/relative_MAE_evol_200steps.pdf\
figures/burgers/relative_MAE_evol_400steps.pdf\
figures/burgers/training/train_length.png\
figures/burgers/training/train_reward.png\
icml2023.bst\
icml2023.sty\
illustrations/advection/fashion_train_illustration.png\
illustrations/advection/mnist_train_illustration.png\
illustrations/advection/mnist_vortex_illustration.png\
illustrations/burgers/burgers_expl1.png\
illustrations/burgers/burgers_expl2.png\
illustrations/IRCNN_compr.drawio.png\
illustrations/SMARL_final.drawio.png\
Makefile\
refs.bib\
