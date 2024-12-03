\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{myproject}[2024/10/07 Modified article class]

\LoadClass[11pt]{article}

% PACKAGES
\RequirePackage[
    letterpaper,
    top = 1in,
    bottom = 1in,
    % left = 0.5in,
    left = 1in,
    % right = 1.5in,
    right = 1in,
    headsep = 1cm
    ]{geometry}                 % document setup
\RequirePackage{amsmath}
\RequirePackage{amssymb}
\RequirePackage{amsthm}
\RequirePackage{fancyhdr}       % provides header and footer
\RequirePackage{verbatim}       % allows for multi-line comments
\RequirePackage{caption}        % better caption options
\RequirePackage{graphicx}       % inserting pictures/figures
\RequirePackage{enumitem}       % customization of list env.
\RequirePackage{siunitx}        % provides scientific notation
\RequirePackage{tabularx}       % tables!!
\RequirePackage{lipsum}         % testing purposes
\RequirePackage{marginnote}     % allows for margin notes
\RequirePackage[
    hidelinks
    ]{hyperref}                 % document hyperlinks
\RequirePackage{cleveref}       % 'smarter' references
\RequirePackage[
    parfill
]{parskip}
\RequirePackage{matlab-prettifier}
\RequirePackage[
    backend = biber,
    style = apa,
    ]{biblatex}
\RequirePackage{bm}
\RequirePackage{cancel}
\RequirePackage{subcaption}

% COMMANDS
\newcommand{\bolden}[1]{\boldmath\textbf{#1}}
\newcommand{\pde}[2]{\frac{\partial#1}{\partial#2}}
\newcommand{\vnorm}[1]{\|#1\|}
\newcommand{\diff}[2]{\ensuremath{D#1}\big\vert_{#2}}
\newcommand{\eval}[1]{\big\vert_{#1}}

% OPERATORS
\DeclareMathOperator{\C}{\mathbb{C}}
\DeclareMathOperator{\R}{\mathbb{R}}
\DeclareMathOperator{\Q}{\mathbb{Q}}
\DeclareMathOperator{\Z}{\mathbb{Z}}
\DeclareMathOperator{\N}{\mathbb{N}}
\DeclareMathOperator{\spn}{\text{span}}
\DeclareMathOperator{\rng}{\text{range}}

% THEOREMS
\newtheoremstyle{perchance}{10pt}{10pt}{}{}{\bfseries}{}{.5em}{}
\theoremstyle{perchance}
\newtheorem*{remark}{Remark}
