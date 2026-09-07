package main

import (
	"log"

	"daybid-dev-service/resources"

	"github.com/gin-gonic/gin"
	"github.com/joho/godotenv"
)

func main() {
	err := godotenv.Load()
	if err != nil {
		log.Printf("Warning: .env file not loaded: %v", err)
	}

	r := gin.Default()
	baseGroup := r.Group("/api")

	baseGroup.GET("/", func(c *gin.Context) {
		c.JSON(200, gin.H{
			"message": "Hello, World!",
		})
	})

	connectomeGroup := baseGroup.Group("/connectome")
	llmGroup := baseGroup.Group("/llm")

	resources.InitMemoryResource(connectomeGroup)
	resources.InitLLMResource(llmGroup)

	r.Run()
}
