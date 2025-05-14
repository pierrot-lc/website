#include <assert.h>
#include <ctype.h>
#include <stdio.h>

#include "hash.h"

#define TOTAL_TYPES 43

int main() {

  char *node_types[TOTAL_TYPES] = {
      "atx_h1_marker",
      "atx_h2_marker",
      "atx_heading",
      "block_mapping",
      "block_mapping_pair",
      "block_node",
      "block_sequence_item",
      "code_fence_content",
      "code_span",
      "code_span_delimiter",
      "collapsed_reference_link",
      "document",
      "emphasis",
      "emphasis_delimiter",
      "fenced_code_block",
      "full_reference_link",
      "heading_content",
      "html_block",
      "inline",
      "inline_link",
      "key",
      "language",
      "latex_block",
      "latex_display",
      "latex_inline",
      "latex_span_delimiter",
      "link",
      "link_destination",
      "link_label",
      "link_reference_definition",
      "link_text",
      "list",
      "list_item",
      "minus_metadata",
      "paragraph",
      "plain_scalar",
      "section",
      "shortcut_link",
      "stream",
      "strong_emphasis",
      "text",
      "uri_autolink",
      "value",
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
