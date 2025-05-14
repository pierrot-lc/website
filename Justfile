CONTENT_DIR := "website"
BUILD_DIR := "delivery/priv"

build: clean
    nix build ./factory

    cp -r "{{ CONTENT_DIR }}" "{{ BUILD_DIR }}"

    find "{{ BUILD_DIR }}" -name "*md" -type f | while read -r file; do \
        ./result/bin/factory --config "./{{ BUILD_DIR }}/global.yaml" --md "${file}" >"${file%.md}.html"; \
    done

    find "{{ BUILD_DIR }}" -name "*html" -type f | while read -r file; do \
        nix-shell -p nodePackages.prettier --run "prettier -w ${file}"; \
    done

setup: clean
    git clone git@github.com:KaTeX/KaTeX.git
    cd KaTeX; nix-shell -p yarn-berry --run "yarn"; nix-shell -p yarn-berry --run "yarn build"
    cp KaTeX/dist/katex.min.js {{ CONTENT_DIR }}
    cp KaTeX/dist/katex.min.css {{ CONTENT_DIR }}
    cp -r KaTeX/dist/fonts {{ CONTENT_DIR }}

clean:
    rm -rf "{{ BUILD_DIR }}"
    rm -rf "result"
    rm -rf "KaTeX"
