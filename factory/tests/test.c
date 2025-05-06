#include <assert.h>
#include <libgen.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "parsers/markdown.h"
#include "tree.h"
#include "ts_utils.h"
#include "write_html.h"

/* Compare a reference HTML file and one converted from a corresponding
 * markdown reference.
 */
int main(int argc, char *argv[]) {
  char *code_1, *code_2;
  char path[100];
  FILE *file;

  Node *converted_tree;

  if (argc < 3) {
    printf("Usage: %s MD_FILE HTML_FILE\n", argv[0]);
    return -1;
  }

  code_1 = read_file(argv[1]);
  assert(code_1 != NULL);

  // Convert markdown into our tree structure.
  converted_tree = parse_markdown(code_1);
  free(code_1);

  // Convert our tree into HTML.
  sprintf(path, "%s.test", argv[2]);
  file = fopen(path, "w");
  assert(file != NULL);
  write_html(file, converted_tree);
  fclose(file);

  code_1 = read_file(argv[2]);
  code_2 = read_file(path);
  assert(code_1 != NULL);
  assert(code_2 != NULL);

  if (strcmp(code_1, code_2) != 0)
    fprintf(stderr, "%s\n%s\n", code_1, code_2);

  assert(strcmp(code_1, code_2) == 0);

  free(code_2);
  free(code_1);
  remove(path);
  return 0;
}
