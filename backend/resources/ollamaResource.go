package resources

import (
	"bytes"
	"daybid-dev-service/middleware"
	"encoding/json"
	"io"
	"net/http"

	"github.com/gin-gonic/gin"
)

type OllamaResource interface {
	PostEmbed(c *gin.Context)
}

type OllamaResourceImpl struct {
}

type EmbedRequest struct {
	Model string `json:"model"`
	Input string `json:"input"`
}

func NewOllamaResource(r *gin.Engine) *OllamaResourceImpl {
	return &OllamaResourceImpl{}
}

func InitOllamaResource(r *gin.Engine) {
	resource := NewOllamaResource(r)

	group := r.Group("/ollama")

	group.Use(middleware.AuthMiddleware())
	group.POST("/embed", resource.PostEmbed)
}

func (o *OllamaResourceImpl) PostEmbed(c *gin.Context) {
	var body EmbedRequest
	if err := c.BindJSON(&body); err != nil {
		c.JSON(400, gin.H{"error": err.Error()})
		return
	}

	req, err := http.NewRequestWithContext(c.Request.Context(), "POST", "http://localhost:11434/api/embed", nil)
	if err != nil {
		c.JSON(500, gin.H{"error": err.Error()})
		return
	}

	jsonBody, err := json.Marshal(body)
	if err != nil {
		c.JSON(500, gin.H{"error": err.Error()})
		return
	}
	req.Body = io.NopCloser(bytes.NewReader(jsonBody))
	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		c.JSON(500, gin.H{"error": err.Error()})
		return
	}
	defer resp.Body.Close()

	bodyBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		c.JSON(500, gin.H{"error": err.Error()})
		return
	}
	c.Data(200, "application/json", bodyBytes)
}
