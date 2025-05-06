#ifndef MARKDOWN_H
#define MARKDOWN_H

#include "tree.h"

/* Return the destination node for the given label. NULL if search has failed.
 */
Node *search_label_destination(Node *, char *);

/* Main function, used to convert the content of the markdown tree into the
 * given file.
 */
Node *parse_markdown(const char *);

#endif
