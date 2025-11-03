package main

import (
    "fmt"
    "net/http"
    "os"
    "time"
)

func main() {
    url := "http://localhost:8000/health"
    if len(os.Args) > 1 {
        url = os.Args[1]
    }
    client := &http.Client{Timeout: 3 * time.Second}
    resp, err := client.Get(url)
    if err != nil {
        fmt.Fprintf(os.Stderr, "ERROR: %v\n", err)
        os.Exit(2)
    }
    defer resp.Body.Close()
    fmt.Printf("%s -> %d\n", url, resp.StatusCode)
    if resp.StatusCode >= 200 && resp.StatusCode < 400 {
        os.Exit(0)
    }
    os.Exit(1)
}
