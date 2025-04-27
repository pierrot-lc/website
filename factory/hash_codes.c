#include <assert.h>
#include <ctype.h>
#include <stdio.h>

#include "hash.h"

#define TOTAL_TYPES 22

int main() {

  char *node_types[TOTAL_TYPES] = {
      "atx_h1_marker",
      "atx_h2_marker",
      "atx_heading",
      "document",
      "emph_em",
      "emph_strong",
      "emphasis",
      "emphasis_delimiter",
      "full_reference_link",
      "heading_content",
      "inline",
      "inline_link",
      "link",
      "link_destination",
      "link_label",
      "link_reference_destination",
      "link_text",
      "paragraph",
      "section",
      "shortcut_link",
      "text",
      "uri_autolink",
  };

  // Make sure there is no collision.
  for (int i = 0; i < TOTAL_TYPES; i++)
    for (int j = 0; j < TOTAL_TYPES; j++)
      assert(i == j || hash(node_types[i]) != hash(node_types[j]));

  for (int i = 0; i < TOTAL_TYPES; i++) {
    printf("#define HASH_");

    for (int j = 0; node_types[i][j] != '\0'; j++)
      printf("%c", toupper(node_types[i][j]));

    printf(" %u\n", hash(node_types[i]));
  }

  return 0;
}
