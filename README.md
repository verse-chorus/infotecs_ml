# Infotecs ML-engineer intern task
Develop a machine learning system that, based on a list of statically imported libraries of an $\texttt{.exe}$ file, predicts whether this file is malicious.
To complete the task, three samples are provided: training, validation and testing. Selections are presented in the form of tsv files with three columns – $\texttt{is_virus}$ – whether the file is malicious: 1=yes, 0=no; filename – file name for review; $\texttt{libs}$ – comma-separated list of libraries statically imported by this file (we used the LIEF library to obtain the list).
