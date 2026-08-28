#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Verify submissions.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
readonly REPO_ROOT
readonly SUBMISSIONS_DIR="${SCRIPT_DIR}/submissions"
readonly VERIFY_PREFIX="${TMPDIR:-/tmp}/scpn-control-papers."
readonly SOURCE_DATE_EPOCH=1787386155

require_command() {
    local command_name="$1"
    if ! command -v "${command_name}" >/dev/null 2>&1; then
        echo "Error: required command is unavailable: ${command_name}" >&2
        exit 1
    fi
}

cleanup() {
    if [[ -n "${VERIFY_ROOT:-}" && -d "${VERIFY_ROOT}" ]]; then
        case "${VERIFY_ROOT}" in
            "${VERIFY_PREFIX}"*) rm -rf -- "${VERIFY_ROOT}" ;;
            *) echo "Error: refusing to remove unexpected path: ${VERIFY_ROOT}" >&2 ;;
        esac
    fi
}

verify_pdf() {
    local built_pdf="$1"
    local tracked_pdf="$2"
    local extracted_text="$3"

    [[ -s "${built_pdf}" ]] || {
        echo "Error: build did not produce a non-empty PDF: ${built_pdf}" >&2
        return 1
    }
    if ! pdffonts "${built_pdf}" | awk 'NR > 2 && $1 != "" && $6 != "yes" { exit 1 }'; then
        echo "Error: PDF contains a font that is not embedded: ${built_pdf}" >&2
        return 1
    fi
    pdftotext "${built_pdf}" "${extracted_text}"
    [[ -s "${extracted_text}" ]] || {
        echo "Error: PDF has no extractable text: ${built_pdf}" >&2
        return 1
    }
    if rg -n '\?\?|\bTODO\b|\bTBD\b|undefined citation|citation undefined' "${extracted_text}"; then
        echo "Error: PDF text contains an unresolved marker: ${built_pdf}" >&2
        return 1
    fi
    pdftotext "${tracked_pdf}" "${extracted_text}.tracked"
    if ! cmp -s -- "${extracted_text}" "${extracted_text}.tracked"; then
        echo "Error: tracked PDF text differs from the reproducible build: ${tracked_pdf}" >&2
        return 1
    fi
}

build_submission() {
    local source_dir="$1"
    local package_name
    local scratch_dir

    package_name="$(basename -- "${source_dir}")"
    scratch_dir="${VERIFY_ROOT}/${package_name}"
    cp -a -- "${source_dir}" "${scratch_dir}"
    find "${scratch_dir}" -maxdepth 1 -type f \
        \( -name 'manuscript.pdf' -o -name 'manuscript.aux' \
        -o -name 'manuscript.bbl' -o -name 'manuscript.blg' \
        -o -name 'manuscript.log' -o -name 'manuscript.out' \
        -o -name 'manuscriptNotes.bib' \) -delete

    for required_file in README.md BUILD.md CITATION.cff submission_metadata.json references.bib; do
        [[ -f "${source_dir}/${required_file}" ]] || {
            echo "Error: ${package_name} is missing ${required_file}" >&2
            return 1
        }
    done
    cffconvert --validate -i "${source_dir}/CITATION.cff" >/dev/null
    jq empty "${source_dir}/submission_metadata.json"

    (
        cd -- "${scratch_dir}"
        export SOURCE_DATE_EPOCH FORCE_SOURCE_DATE=1
        if [[ -f manuscript.md ]]; then
            pandoc manuscript.md --from=markdown --citeproc \
                --bibliography=references.bib \
                --metadata=author:"Miroslav Šotek" \
                --pdf-engine=pdflatex --output=manuscript.pdf
        elif [[ -f manuscript.tex ]]; then
            pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex >pass1.log
            bibtex manuscript >bibtex.log
            pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex >pass2.log
            pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex >pass3.log
            if rg -n 'undefined citations|undefined references|There were undefined|multiply defined|LaTeX Error' manuscript.log; then
                echo "Error: ${package_name} has unresolved LaTeX output" >&2
                return 1
            fi
        else
            echo "Error: ${package_name} has no supported manuscript source" >&2
            return 1
        fi
        verify_pdf manuscript.pdf "${source_dir}/manuscript.pdf" manuscript.txt
    )

    echo "[OK] ${package_name}"
}

main() {
    local submission_dir
    local found=0
    local tracked_auxiliaries

    for command_name in bibtex cffconvert cmp jq pandoc pdffonts pdflatex pdftotext rg; do
        require_command "${command_name}"
    done

    tracked_auxiliaries="$(git -C "${REPO_ROOT}" ls-files 'papers/**' \
        | rg '\.(aux|bbl|blg|fls|fdb_latexmk|log|out|synctex\.gz|toc)$|Notes\.bib$' || true)"
    if [[ -n "${tracked_auxiliaries}" ]]; then
        echo "Error: disposable manuscript build products are tracked:" >&2
        echo "${tracked_auxiliaries}" >&2
        exit 1
    fi

    VERIFY_ROOT="$(mktemp -d "${VERIFY_PREFIX}XXXXXX")"
    export VERIFY_ROOT
    trap cleanup EXIT INT TERM

    while IFS= read -r -d '' submission_dir; do
        found=1
        build_submission "${submission_dir}"
    done < <(find "${SUBMISSIONS_DIR}" -mindepth 1 -maxdepth 1 -type d \
        -name '[0-9][0-9][0-9]_*' -print0 | sort -z)

    if [[ "${found}" -eq 0 ]]; then
        echo "Error: no numbered submission packages found" >&2
        exit 1
    fi

    echo "All canonical submission packages passed reproducible build checks."
}

main "$@"
