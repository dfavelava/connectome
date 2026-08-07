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
		log.Fatal("Error loading .env file")
	}

	r := gin.Default()

	r.GET("/", func(c *gin.Context) {
		c.JSON(200, gin.H{
			"message": "Hello, World!",
		})
	})

	resources.InitOllamaResource(r)
	resources.InitAWSResource(r)

	r.Run()
}
