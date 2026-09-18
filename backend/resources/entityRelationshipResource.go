package resources

import (
	"net/http"
	"slices"

	"github.com/gin-gonic/gin"

	"connectome-dev-service/managers"
	"connectome-dev-service/middleware"
)

// relationshipKinds mirrors RelationshipKind in
// connectomeMCP/src/connectomemcp/server.py.
var relationshipKinds = []string{"fact", "hypothesis", "rumor"}

// defaultEntityRelationshipKind mirrors DEFAULT_RELATIONSHIP_KIND.
const defaultEntityRelationshipKind = "fact"

// EntityRelationshipRequest names one relationship to assert between two
// entities. Kind carries the same truth-status vocabulary as a memory
// document's relationship entries (see relationshipDocument in
// relationshipPatch.go) for shape parity with that model, but - like
// connectomemcp.server's merge_member_of - this endpoint doesn't gate the
// member_of merge on it; it's accepted and validated so a future caller that
// also wants to persist the relationship claim itself can reuse this same
// request shape without a breaking change.
type EntityRelationshipRequest struct {
	SubjectEntityID string         `json:"subjectEntityId"`
	Predicate       string         `json:"predicate"`
	ObjectEntityID  *string        `json:"objectEntityId"`
	Kind            string         `json:"kind"`
	SubjectKind     *string        `json:"subjectKind"`
	SubjectMeta     map[string]any `json:"subjectMeta"`
	Tome            string         `json:"tome"`
}

// EntityRelationshipResponse returns the entity records affected by an
// EntityRelationshipRequest, so the caller can see what changed.
type EntityRelationshipResponse struct {
	Subject EntityWithMemories  `json:"subject"`
	Object  *EntityWithMemories `json:"object,omitempty"`
}

type EntityResourceImpl struct {
	manager managers.MemoryManager
}

func NewEntityResource(manager managers.MemoryManager) *EntityResourceImpl {
	return &EntityResourceImpl{manager: manager}
}

func InitEntityResource(r *gin.RouterGroup, manager managers.MemoryManager) {
	resource := NewEntityResource(manager)

	group := r.Group("/entity")
	group.Use(middleware.AuthMiddleware())
	group.POST("/relationship", resource.upsertRelationship)
}

// upsertRelationship handles POST /entity/relationship: given
// {subjectEntityId, predicate, objectEntityId, kind, subjectKind,
// subjectMeta}, it upserts stub ent_<id>.json records for subject/object
// entities that don't have one yet, for the member_of predicate merges
// objectEntityId into the subject entity's member_of list, and - when given -
// merges subjectKind/subjectMeta onto the subject entity. See
// UpsertEntityRelationship.
func (resource *EntityResourceImpl) upsertRelationship(c *gin.Context) {
	var req EntityRelationshipRequest
	if err := c.BindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	if req.SubjectEntityID == "" || req.Predicate == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "subjectEntityId and predicate are required"})
		return
	}
	if req.Kind == "" {
		req.Kind = defaultEntityRelationshipKind
	} else if !slices.Contains(relationshipKinds, req.Kind) {
		c.JSON(http.StatusBadRequest, gin.H{"error": "kind must be one of: fact, hypothesis, rumor"})
		return
	}

	subject, object, err := UpsertEntityRelationship(resource.manager, req.Tome, req.SubjectEntityID, req.Predicate, req.ObjectEntityID, req.SubjectKind, req.SubjectMeta)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, EntityRelationshipResponse{Subject: subject, Object: object})
}
