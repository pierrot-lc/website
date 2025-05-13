CONTENT_DIR := "website"
BUILD_DIR := "delivery/priv"

build: clean
    nix build ./factory

    cp -r "{{ CONTENT_DIR }}" "{{ BUILD_DIR }}"

    find "{{ BUILD_DIR }}" -name "*md" -type f | while read -r file; do \
        ./result/bin/factory --config "./{{ BUILD_DIR }}/global.yaml" --md "${file}" >"${file%.md}.html"; \
    done

clean:
    rm -rf "{{ BUILD_DIR }}"
    rm -rf "result"
