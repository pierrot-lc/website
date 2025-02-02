#include "hash.h"
#include <ctype.h>
#include <stdio.h>

int main() {
  char *node_types[5] = {
      "content", "heading", "section", "source_file", "text",
  };

  for (int i = 0; i < 5; i++) {
    printf("#define HASH_");

    for (int j = 0; node_types[i][j] != '\0'; j++)
      printf("%c", toupper(node_types[i][j]));

    printf(" %lu\n", hash(node_types[i]));
  }

  return 0;
}
