package resources

import (
	"errors"
	"net/http"

	"github.com/gin-gonic/gin"

	"connectome-dev-service/managers"
	"connectome-dev-service/middleware"
)

type TomeResourceImpl struct {
	manager    managers.MemoryManager
	embeddings TomeEmbeddingsDeleter
}

func NewTomeResource(manager managers.MemoryManager, embeddings TomeEmbeddingsDeleter) *TomeResourceImpl {
	return &TomeResourceImpl{manager: manager, embeddings: embeddings}
}

func InitTomeResource(r *gin.RouterGroup, manager managers.MemoryManager, embeddings TomeEmbeddingsDeleter) {
	resource := NewTomeResource(manager, embeddings)

	group := r.Group("/tome")
	group.Use(middleware.AuthMiddleware())
	group.DELETE("/:id", resource.destroy)
}

// destroy handles DELETE /tome/:id?confirm=true: destroys every blob and
// embeddings row scoped to the given tome id, subject to DestroyTome's
// guard - see checkTomeDestroyAllowed.
func (resource *TomeResourceImpl) destroy(c *gin.Context) {
	tome := c.Param("id")
	confirm := c.Query("confirm") == "true"

	err := DestroyTome(c.Request.Context(), resource.manager, resource.embeddings, tome, confirm)
	if err != nil {
		if errors.Is(err, ErrDestroyDefaultTome) || errors.Is(err, ErrDestroyNeedsConfirm) {
			c.JSON(http.StatusForbidden, gin.H{"error": err.Error()})
			return
		}
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{"message": "destroyed", "tome": tome})
}
