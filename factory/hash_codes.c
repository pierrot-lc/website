#include "hash.h"
#include <ctype.h>
#include <stdio.h>

int main() {
  char *node_types[10] = {
      "atx_h1_marker",   "atx_h2_marker", "atx_heading",
      "document",        "emphasis",      "emphasis_delimiter",
      "heading_content", "inline",        "paragraph",
      "section",
  };

  for (int i = 0; i < 9; i++) {
    printf("#define HASH_");

    for (int j = 0; node_types[i][j] != '\0'; j++)
      printf("%c", toupper(node_types[i][j]));

    printf(" %u\n", hash(node_types[i]));
  }

  return 0;
}
