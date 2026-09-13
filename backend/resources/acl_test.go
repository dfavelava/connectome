package resources

import (
	"reflect"
	"testing"
)

func TestResolveACLUsesExplicitValueWhenPresent(t *testing.T) {
	explicit := []string{"GM"}
	if got := ResolveACL(&explicit); !reflect.DeepEqual(got, explicit) {
		t.Fatalf("expected explicit acl %v, got %v", explicit, got)
	}

	explicitEmpty := []string{}
	if got := ResolveACL(&explicitEmpty); !reflect.DeepEqual(got, explicitEmpty) {
		t.Fatalf("expected explicit empty acl to be preserved, got %v", got)
	}
}

func TestResolveACLFallsBackToDefaultACLEnvWhenNil(t *testing.T) {
	t.Setenv("DEFAULT_ACL", "GM, players")
	if got := ResolveACL(nil); !reflect.DeepEqual(got, []string{"GM", "players"}) {
		t.Fatalf("expected DEFAULT_ACL to split into [GM players], got %v", got)
	}
}

func TestResolveACLIsUnrestrictedWhenNilAndUnconfigured(t *testing.T) {
	t.Setenv("DEFAULT_ACL", "")
	if got := ResolveACL(nil); len(got) != 0 {
		t.Fatalf("expected unrestricted (empty) acl when DEFAULT_ACL is unset, got %v", got)
	}
}
