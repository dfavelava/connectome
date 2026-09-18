package main

import (
	"log"

	"connectome-dev-service/daos"
	"connectome-dev-service/managers"
	"connectome-dev-service/resources"

	"github.com/gin-gonic/gin"
	"github.com/joho/godotenv"
)

func main() {
	err := godotenv.Load()
	if err != nil {
		log.Printf("Warning: .env file not loaded: %v", err)
	}

	postgresManager := managers.NewPostgresManager()
	embeddingsDao := daos.NewEmbeddingsDao(postgresManager.Pool)
	ollamaManager := managers.NewOllamaManager()
	memoryManager := managers.NewMemoryManagerFromEnv()

	r := gin.Default()
	baseGroup := r.Group("/api")

	baseGroup.GET("/", func(c *gin.Context) {
		c.JSON(200, gin.H{
			"message": "Hello, World!",
		})
	})

	connectomeGroup := baseGroup.Group("/connectome")
	llmGroup := baseGroup.Group("/llm")

	resources.InitMemoryResource(connectomeGroup, memoryManager, ollamaManager, embeddingsDao)
	resources.InitSearchResource(connectomeGroup, memoryManager, ollamaManager, embeddingsDao)
	resources.InitEntityResource(connectomeGroup, memoryManager)
	resources.InitTomeResource(connectomeGroup, memoryManager, embeddingsDao)
	resources.InitLLMResource(llmGroup)

	r.Run()
}
