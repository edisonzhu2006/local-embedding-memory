%%
%% This is file `puthesis_undergraduate.cls',
%% Princeton ORFE/SEAS undergraduate thesis class (E. Papageorgiou).
%% Used here for ECE 299 Sophomore IW report.
%%
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{puthesis_undergraduate}
    [2006/11/10 v1.0 Princeton University Undergraduate Thesis class]
\RequirePackage{setspace}
\newcounter{subyear}
\setcounter{subyear}{\number\year}
\def\submitted#1{\gdef\@submitted{#1}}
\def\advisor#1{\gdef\@advisor{#1}}
\def\@submittedyear{\ifnum\month>10 \stepcounter{subyear}\thesubyear
  \else\thesubyear\fi}
\def\@submittedmonth{\ifnum\month>10 January\else\ifnum\month>5 November
  \else June\fi\fi}
\long\def\@abstract{\@latex@error{No \noexpand\abstract given}\@ehc}
\newcommand*{\frontmatter}{
  \pagenumbering{roman}
  }
\newcommand*{\mainmatter}{\pagenumbering{arabic}}
\newcommand*{\makelot}{}
\newcommand*{\makelof}{}
\newcommand*{\makelos}{}
\newcommand*{\begincmd}{
\doublespacing
\frontmatter\maketitlepage\makestatements\makeabstract
\makeacknowledgements\makededication\tableofcontents\clearpage
\makelot\clearpage\makelof\clearpage\makelos
\clearpage\mainmatter}
\def\@submitted{\@submittedmonth~\@submittedyear}
\def\@dept{Electrical and Computer Engineering}
\def\@deptpref{Department of}
\def\departmentprefix#1{\gdef\@deptpref{#1}}
\def\department#1{\gdef\@dept{#1}}
\long\def\acknowledgements#1{\gdef\@acknowledgements{#1}}
\def\dedication#1{\gdef\@dedication{#1}}
\newcommand{\maketitlepage}{{
  \thispagestyle{empty}
  \sc
  \vspace*{0in}
  \begin{center}
    \LARGE \@title
  \end{center}
  \vspace{.6in}
  \begin{center}
   \@author \\
    Advisor: \@advisor
  \end{center}
  \vspace{.6in}
  \begin{center}
Bachelor of Science in Engineering\\
\@deptpref{} \@dept\\
Princeton University
  \end{center}
  \vspace{0.9in}
  \begin{center}
    \@submitted
  \end{center}
  \clearpage
  }}
\newcommand*{\makestatements}{
  \thispagestyle{empty}
  \vspace*{.2in}
\noindent I hereby declare that I am the sole author of this thesis.\\ \\
\noindent I authorize Princeton University to lend this thesis to other
    institutions or individuals for the purpose of scholarly research.\vspace*{.5in}\\ \\
  \hspace*{3.5in}
  \makebox[2.0in]{\hrulefill} \\
  \hspace*{3.5in}
  \@author

   \vspace*{0.8in}
\noindent I further authorize Princeton University to reproduce this thesis by photocopying or by
other means, in total or in part, at the request of other institutions or individuals for the
purpose of scholarly research.\vspace*{.5in}\\ \\
  \hspace*{3.5in}
  \makebox[2.0in]{\hrulefill} \\
  \hspace*{3.5in}
  \@author
  \clearpage}
\newcommand*{\makeabstract}{
  \newpage
  \phantomsection
  \addcontentsline{toc}{section}{Abstract}
  \begin{center}
  \Large \textbf{Abstract}
  \end{center}
  \@abstract
  \clearpage
  }
\def\makeacknowledgements{
  \ifx\@acknowledgements\undefined
  \else
    \phantomsection
    \addcontentsline{toc}{section}{Acknowledgements}
    \begin{center}
      \Large \textbf{Acknowledgements}
    \end{center}
    \@acknowledgements
    \clearpage
  \fi
  }
\def\makededication{
  \ifx\@dedication\undefined
  \else
    \vspace*{1.5in}
    \begin{flushright}
      \@dedication
      for posterity
    \end{flushright}
    \clearpage
  \fi
  }
\DeclareOption{myorder}{
  \renewcommand*{\begincmd}{\doublespacing}}
\DeclareOption{lot}{\renewcommand*{\makelot}{
  \addcontentsline{toc}{section}{List of Tables}\listoftables}}
\DeclareOption{lof}{\renewcommand*{\makelof}{
  \addcontentsline{toc}{section}{List of Figures}\listoffigures}}
\DeclareOption{los}{
  \renewcommand*{\makelos}{
    \RequirePackage{losymbol}
    \section*{List of Symbols\@mkboth {LIST OF SYMBOLS}{LIST OF SYMBOLS}}
    \@starttoc{los}
    \addcontentsline{toc}{section}{List of Symbols}
  }
}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{report}}
\ProcessOptions
\LoadClass[12pt]{report}
\setlength{\oddsidemargin}{0.25in}
\setlength{\evensidemargin}{.25in}
\setlength{\topmargin}{-.5in}
\setlength{\textheight}{9in}
\setlength{\textwidth}{6.0in}
\long\def\abstract#1{\gdef\@abstract{#1}}
\AtBeginDocument{\begincmd}
\endinput
