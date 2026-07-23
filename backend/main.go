package main

import (
	"bytes"
	"encoding/json"
	"io"
	"log"
	"net/http"

	"daybid-dev-service/middleware"

	"github.com/gin-gonic/gin"
	"github.com/joho/godotenv"
)

type EmbedRequest struct {
	Model string `json:"model"`
	Input string `json:"input"`
}

func main() {
	err := godotenv.Load()
	if err != nil {
		log.Fatal("Error loading .env file")
	}

	r := gin.Default()
	r.GET("/", func(c *gin.Context) {
		c.JSON(200, gin.H{
			"message": "Hello, World!",
		})
	})

	protected := r.Group("/ollama")

	protected.Use(middleware.AuthMiddleware())
	{
		protected.POST("/embed", func(c *gin.Context) {
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
		})
	}

	r.Run()
}
