package resources

import (
	"os"
	"strings"
)

// ResolveACL determines the acl to store for a memory. A non-nil acl (the
// memory's frontmatter explicitly set one, even to an empty list) is used
// as-is. A nil acl (the frontmatter omitted the field entirely) picks up
// this instance's configured DEFAULT_ACL, so every writer - MCP, and any
// future client - gets the same default policy without reimplementing it.
func ResolveACL(acl *[]string) []string {
	if acl != nil {
		return *acl
	}
	return defaultACLFromEnv()
}

// defaultACLFromEnv reads DEFAULT_ACL as a comma-separated list of
// entity/group ids. Unset or blank yields an empty (unrestricted) acl - the
// sane default for a generic Connectome instance with no configured policy.
func defaultACLFromEnv() []string {
	raw := strings.TrimSpace(os.Getenv("DEFAULT_ACL"))
	if raw == "" {
		return []string{}
	}

	parts := strings.Split(raw, ",")
	acl := make([]string, 0, len(parts))
	for _, p := range parts {
		if p = strings.TrimSpace(p); p != "" {
			acl = append(acl, p)
		}
	}
	return acl
}
