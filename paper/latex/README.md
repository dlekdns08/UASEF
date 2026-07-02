# UASEF Round 9 — LaTeX Build

PDF 빌드용 LaTeX 소스. 본 문서는 [results/round9/ROUND9_FINAL_REPORT.md](../../results/round9/ROUND9_FINAL_REPORT.md) 의 핵심 내용을 학술 논문 형식으로 압축한 단일 self-contained TeX 파일입니다.

## 파일

- `uasef_round9.tex` — main paper (영문), thebibliography 통합 (별도 .bib 파일 불필요)
- `README.md` — 본 문서

## 빌드 방법

### Option 1: pdflatex (수동, 3-pass)

```bash
cd paper/latex
pdflatex uasef_round9.tex      # 1st pass — aux 생성
pdflatex uasef_round9.tex      # 2nd pass — references 해결
pdflatex uasef_round9.tex      # 3rd pass — TOC 등 finalize
```

### Option 2: latexmk (자동, 권장)

```bash
cd paper/latex
latexmk -pdf uasef_round9.tex
```

### Option 3: Docker (재현 가능 빌드)

```bash
cd paper/latex
docker run --rm -v "$PWD":/work -w /work texlive/texlive:latest \
    latexmk -pdf uasef_round9.tex
```

### Option 4: macOS (MacTeX)

```bash
brew install --cask mactex   # 한 번만
cd paper/latex
latexmk -pdf uasef_round9.tex
```

## 의존 패키지

`uasef_round9.tex` 가 사용하는 LaTeX 패키지 (모두 TeX Live 표준):
- `geometry`, `amsmath`, `amssymb`, `amsthm`, `booktabs`, `array`, `longtable`
- `hyperref`, `xcolor`, `fancyvrb`, `soul`, `microtype`, `caption`, `enumitem`

## 빌드 산출물

성공 시 `uasef_round9.pdf` (예상 약 8–12 페이지) 가 생성됩니다.

## 정리

```bash
latexmk -c            # aux, log 등 중간 산출 정리
latexmk -C            # PDF 까지 정리
```

## 출처 동기화

본 LaTeX 는 다음 Markdown 소스에서 derive:

| TeX 섹션 | Markdown 출처 |
|---|---|
| Abstract | [ROUND9_FINAL_REPORT.md#abstract](../../results/round9/ROUND9_FINAL_REPORT.md) |
| §1–§3 | [ROUND9_FINAL_REPORT.md §1–§3](../../results/round9/ROUND9_FINAL_REPORT.md) |
| §4–§5 Methods/Setup | [ROUND9_FINAL_REPORT.md §4–§5](../../results/round9/ROUND9_FINAL_REPORT.md) |
| §6 Results | 실험 산출 JSON 직접 인용 (`results/round9/*.json`) |
| §7 Discussion | [ROUND9_FINAL_REPORT.md §7](../../results/round9/ROUND9_FINAL_REPORT.md) |
| §8 Limitations | [ROUND9_FINAL_REPORT.md §8](../../results/round9/ROUND9_FINAL_REPORT.md) |
| References | [ROUND9_FINAL_REPORT.md §References](../../results/round9/ROUND9_FINAL_REPORT.md) (27 entries) |

Markdown 보고서 update 시 LaTeX 도 수동 동기화 필요.

## 한국어 버전

영문 LaTeX 만 제공. 한국어 버전은 [ROUND9_FINAL_REPORT_KO.md](../../results/round9/ROUND9_FINAL_REPORT_KO.md) 참조. 한국어 LaTeX 가 필요하면 `kotex` 패키지 추가하고 별도 빌드 (`uasef_round9_ko.tex` 작성 필요 — 본 릴리스에는 미포함).
