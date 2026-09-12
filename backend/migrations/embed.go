// Package migrations embeds the SQL migration files so they ship inside the
// backend binary and don't need to be mounted separately in production images.
package migrations

import "embed"

//go:embed *.sql
var FS embed.FS
