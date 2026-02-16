import os

def read_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

# Read actual source code
code_validator = read_file('src/models/validator.py')
code_api = read_file('src/api.py')
code_predictor = read_file('src/models/predictor.py')
code_drift = read_file('src/models/drift.py')
code_recovery = read_file('src/data/recovery.py')
code_pipeline = read_file('src/data/pipeline.py')
code_consensus = read_file('src/utils/consensus.py')
code_safety = read_file('src/utils/safety.py')
code_circuit = read_file('src/utils/circuit_breaker.py')
code_docker = read_file('Dockerfile')

# Define LaTeX Structure and Narratives
latex_content = r"""\documentclass[12pt,a4paper,oneside]{book}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage[spanish, es-noshorthands]{babel}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{amsfonts}
\usepackage{graphicx}
\usepackage{xcolor}
\usepackage{listings}
\usepackage{algorithm}
\usepackage{algpseudocode}
\usepackage{geometry}
\usepackage{fancyhdr}
\usepackage{setspace}

\geometry{left=2.5cm, right=2.5cm, top=2.5cm, bottom=2.5cm}
\onehalfspacing

\definecolor{codegreen}{rgb}{0,0.6,0}
\definecolor{codegray}{rgb}{0.5,0.5,0.5}
\definecolor{codepurple}{rgb}{0.58,0,0.82}
\definecolor{backcolour}{rgb}{0.95,0.95,0.95}

\lstdefinestyle{pythonStyle}{
    backgroundcolor=\color{backcolour},
    commentstyle=\color{codegreen},
    keywordstyle=\color{codepurple},
    numberstyle=\tiny\color{codegray},
    stringstyle=\color{codegreen},
    basicstyle=\ttfamily\footnotesize,
    breakatwhitespace=false,
    breaklines=true,
    captionpos=b,
    keepspaces=true,
    numbers=left,
    numbersep=5pt,
    showspaces=false,
    showstringspaces=false,
    showtabs=false,
    tabsize=4,
    frame=single,
    rulecolor=\color{codegray},
}

\lstset{style=pythonStyle}

\title{\textbf{Arquitectura Integral para Sistemas de Predicción Financiera}}
\author{BioPhys-Tech Lab}
\date{\today}

\begin{document}

\maketitle
\tableofcontents

\chapter{Introducción}
Este documento detalla la implementación técnica del sistema de predicción financiera en tiempo real.

\chapter{Solución ML Engineer}

\section{Validación de Datos (Pydantic)}
\begin{lstlisting}[language=Python, caption=Validadores Pydantic (src/models/validator.py)]
""" + code_validator + r"""
\end{lstlisting}

\section{Motor de Inferencia Optimizado}
\begin{lstlisting}[language=Python, caption=Predictor de Baja Latencia (src/models/predictor.py)]
""" + code_predictor + r"""
\end{lstlisting}

\section{Detección de Drift}
\begin{lstlisting}[language=Python, caption=Drift Detector con KS Test (src/models/drift.py)]
""" + code_drift + r"""
\end{lstlisting}

\section{API y Monitorización}
\begin{lstlisting}[language=Python, caption=FastAPI App (src/api.py)]
""" + code_api + r"""
\end{lstlisting}

\section{Containerización}
\begin{lstlisting}[language=bash, caption=Dockerfile Multi-stage]
""" + code_docker + r"""
\end{lstlisting}

\chapter{Solución Data Engineer}

\section{Pipeline de Recuperación de Datos}
\begin{lstlisting}[language=Python, caption=Data Gap Recovery Engine (src/data/recovery.py)]
""" + code_recovery + r"""
\end{lstlisting}

\section{Pipeline de Ingesta}
\begin{lstlisting}[language=Python, caption=Ingestion Pipeline & Simulator (src/data/pipeline.py)]
""" + code_pipeline + r"""
\end{lstlisting}

\section{Resiliencia: Circuit Breaker}
\begin{lstlisting}[language=Python, caption=Circuit Breaker Pattern (src/utils/circuit_breaker.py)]
""" + code_circuit + r"""
\end{lstlisting}

\chapter{Bonus: Seguridad y Consenso}

\section{Safety Breaker System}
\begin{lstlisting}[language=Python, caption=Sistema de Seguridad de Mercado (src/utils/safety.py)]
""" + code_safety + r"""
\end{lstlisting}

\section{Consenso Multi-Proveedor}
\begin{lstlisting}[language=Python, caption=Algoritmo de Consenso (src/utils/consensus.py)]
""" + code_consensus + r"""
\end{lstlisting}

\chapter{Conclusión}
El sistema ha sido implementado siguiendo estrictamente los requerimientos de latencia, seguridad y resiliencia.

\end{document}
"""

with open('collaboration.tex', 'w', encoding='utf-8') as f:
    f.write(latex_content)

print(f"Successfully generated collaboration.tex with {len(latex_content)} chars")
