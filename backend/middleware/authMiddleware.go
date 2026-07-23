package middleware

import (
	"strings"

	"github.com/gin-gonic/gin"
)

func authMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		token := c.GetHeader("Authorization")
		if token == "" {
			c.JSON(401, gin.H{"error": "Authorization header is required"})
			c.Abort()
			return
		}

		parts := strings.SplitN(token, " ", 2)

		if len(parts) != 2 || parts[0] != "Bearer" {
			c.JSON(401, gin.H{"error": "Invalid token"})
			c.Abort()
			return
		}

		key := parts[1]

		if key != getAllowedKey() {
			c.JSON(401, gin.H{"error": "Invalid token"})
			c.Abort()
			return
		}

		c.Next()
	}
}

func getAllowedKey() string {
	return ""
}
