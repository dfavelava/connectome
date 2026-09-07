package resources

import (
	"daybid-dev-service/managers"
	"daybid-dev-service/middleware"

	"github.com/gin-gonic/gin"
)

type LLMResourceImpl struct {
	manager *managers.OllamaManager
}

type RequestWithInput struct {
	Input string `json:"input"`
}

type EmbeddingResponse struct {
	Embeddings []float32 `json:"embeddings"`
}

func NewLLMResource() *LLMResourceImpl {
	manager := managers.NewOllamaManager()
	return &LLMResourceImpl{manager: manager}
}

func InitLLMResource(r *gin.RouterGroup) {
	resource := NewLLMResource()

	group := r.Group("/")
	group.Use(middleware.AuthMiddleware())

	group.POST("/embed", resource.embed)
}

func (r *LLMResourceImpl) embed(c *gin.Context) {
	var req RequestWithInput
	if err := c.BindJSON(&req); err != nil {
		c.JSON(400, gin.H{"error": err.Error()})
		return
	}

	embeddings, err := r.manager.Embed(req.Input)
	if err != nil {
		c.JSON(500, gin.H{"error": err.Error()})
		return
	}

	c.JSON(200, EmbeddingResponse{Embeddings: embeddings})
}
