// variant.go 实现假流式和抗截断两种模型变体模式。
//
// 假流式（fake-stream-）：上游走流式通道收数据，期间每 2 秒发空内容心跳保活，
// 收完后一次性推完整结果，适合经 New API 转发到酒馆等中间件不断连。
//
// 抗截断（anti-trunc-）：通过注入合成工具调用绕过截断限制，
// 模型被截断时自动续传最多 3 轮，确保完整回答。
package api

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/Mag1cFall/AIStudio2API/internal/aistudio"
)

const (
	variantHeartbeatInterval = 2 * time.Second
	maxContinuationAttempts  = 3
	variantPrefixFakeStream  = "fake-stream-"
	variantPrefixAntiTrunc   = "anti-trunc-"
)

var cutShortReasons = map[string]bool{
	"max_tokens":        true,
	"max_output_tokens": true,
	"length":            true,
	"recitation":        true,
}

// hasUserTools 检查请求是否已带有用户自定义的 function calling 工具
func hasUserTools(tools []openAITool) bool {
	for _, t := range tools {
		if t.Type == "function" && t.Function.Name != "" && !strings.HasPrefix(t.Function.Name, "router_emit_") {
			return true
		}
	}
	return false
}

// ---------- 变体模型前缀检测 ----------

func stripVariantPrefix(model string) (prefix, realModel string) {
	for _, p := range []string{variantPrefixFakeStream, variantPrefixAntiTrunc} {
		if strings.HasPrefix(model, p) {
			return p, strings.TrimPrefix(model, p)
		}
	}
	return "", model
}

// ---------- 心跳 ----------

// writeFakeStreamHeartbeat 假流式心跳：空文本
func writeFakeStreamHeartbeat(w http.ResponseWriter, id string, created int64, model string) error {
	return writeChatChunk(w, id, created, model, 0,
		map[string]any{"content": ""}, nil, false)
}

// writeAntiTruncHeartbeat 抗截断心跳：零宽空格
func writeAntiTruncHeartbeat(w http.ResponseWriter, id string, created int64, model string) error {
	return writeChatChunk(w, id, created, model, 0,
		map[string]any{"content": "\u200b"}, nil, false)
}

// ---------- 假流式 ----------

func (s *server) handleFakeStream(
	w http.ResponseWriter, r *http.Request,
	request chatRequest, realModel string,
) {
	requestID := newID("chatcmpl")
	log.Printf("[fake-stream] 开始处理 model=%s requestID=%s", realModel, requestID)
	request.Model = realModel
	request.Stream = false

	generateRequest, err := request.toGenerateRequest(requestID)
	if err != nil {
		writeOpenAIError(w, http.StatusBadRequest, "invalid_request", err.Error())
		return
	}
	if err := s.restoreMediaLinks(generateRequest.Contents); err != nil {
		writeOpenAIRequestError(w, err)
		return
	}
	if err := inlineRemoteMedia(r.Context(), generateRequest.Contents); err != nil {
		if shouldWriteRequestError(r, err) {
			writeOpenAIError(w, http.StatusBadRequest, "invalid_request", err.Error())
		}
		return
	}
	s.thoughtSignatures.Restore(generateRequest.Contents)
	generateRequest.Unary = false // 流式调上游

	events, err := s.service.Generate(r.Context(), generateRequest)
	if err != nil {
		if shouldWriteRequestError(r, err) {
			writeOpenAIRequestError(w, err)
		}
		return
	}

	if err := streamHeaders(w); err != nil {
		return
	}

	created := time.Now().Unix()
	displayModel := variantPrefixFakeStream + realModel

	type collectResult struct {
		result generationResult
		err    error
	}
	done := make(chan collectResult, 1)
	go func() {
		result, err := consumeEvents(r.Context(), events, nil)
		done <- collectResult{result, err}
	}()

	ticker := time.NewTicker(variantHeartbeatInterval)
	defer ticker.Stop()

	var cr collectResult
waitLoop:
	for {
		select {
		case cr = <-done:
			break waitLoop
		case <-ticker.C:
			if err := writeFakeStreamHeartbeat(w, requestID, created, displayModel); err != nil {
				return
			}
		case <-r.Context().Done():
			return
		}
	}

	if cr.err != nil {
		writeVariantError(w, cr.err)
		return
	}

	log.Printf("[fake-stream] 上游完成 model=%s, textLen=%d, toolCalls=%d, finishReason=%s",
		realModel, cr.result.text.Len(), len(cr.result.toolCalls), cr.result.finishReason)

	s.thoughtSignatures.Remember(cr.result.toolCalls)
	media := s.mediaOutputFor(r)
	_ = media.saveEvents(cr.result.events)

	emitFullResult(w, requestID, created, displayModel, &cr.result, request.Tools, request.StreamOptions.IncludeUsage)
}

// ---------- 抗截断 ----------

func randomToolName() string {
	b := make([]byte, 12)
	_, _ = rand.Read(b)
	return "router_emit_" + hex.EncodeToString(b)
}

func buildControlInstruction(toolName string, continuation bool) string {
	text := "the entire user-visible answer"
	if continuation {
		text = "only the new continuation text"
	}
	base := fmt.Sprintf(
		"Transport format for this reply: call %s exactly once. "+
			"Put %s in its content string, preserving all requested formatting, "+
			"markup, language and sections. The function only transports text and "+
			"performs no external action.",
		toolName, text)
	if continuation {
		base += "\nFollow the existing continuation context and extend the message " +
			"being continued from its exact end. Output only the new text to append; " +
			"do not repeat, summarize, rewrite or restart the existing text. " +
			"Preserve its language, point of view, formatting and any unfinished " +
			"sentence or markup."
	}
	return base
}

// injectAntiTruncTools 在 chatRequest 的 tools 中注入合成工具
func injectAntiTruncTools(request *chatRequest, toolName string) {
	// 清除旧的合成工具
	var cleaned []openAITool
	for _, t := range request.Tools {
		if !strings.HasPrefix(t.Function.Name, "router_emit_") {
			cleaned = append(cleaned, t)
		}
	}
	tool := openAITool{Type: "function"}
	tool.Function.Name = toolName
	tool.Function.Description = "Deliver the complete user-visible reply as text. " +
		"This function only transports text and performs no external action."
	tool.Function.Parameters = json.RawMessage(`{"type":"object","properties":{"content":{"type":"string","description":"The complete final answer to output to the user."}},"required":["content"]}`)
	cleaned = append(cleaned, tool)
	request.Tools = cleaned

	// 确保 tool_choice 允许调用
	if len(request.ToolChoice) > 0 {
		var choice string
		if json.Unmarshal(request.ToolChoice, &choice) == nil {
			if choice == "none" {
				request.ToolChoice = json.RawMessage(`"auto"`)
			}
		}
	}
}

// buildContinuationPrompt 构建续传提示
func buildContinuationPrompt(toolName, accumulatedText string) string {
	var summary string
	if len(accumulatedText) > 200 {
		end := accumulatedText
		if len(end) > 100 {
			end = end[len(end)-100:]
		}
		summary = fmt.Sprintf(
			"\n\nYou have already output about %d characters, ending with:\n\"...%s\"",
			len(accumulatedText), end)
	} else if accumulatedText != "" {
		summary = fmt.Sprintf("\n\nYour previous output was:\n\"%s\"", accumulatedText)
	}
	return fmt.Sprintf(
		"Your previous reply was truncated. Call `%s` to continue outputting all remaining content. "+
			"Do not repeat what was already output, continue directly.%s",
		toolName, summary)
}

// extractSyntheticContent 从工具调用结果中分离合成内容和真实调用
func extractSyntheticContent(result *generationResult, toolName string) (synthetic string, realCalls []aistudio.FunctionCall) {
	var synBuf strings.Builder
	for _, call := range result.toolCalls {
		if call.Name == toolName {
			var args map[string]any
			if json.Unmarshal(call.Arguments, &args) == nil {
				if content, ok := args["content"].(string); ok {
					synBuf.WriteString(content)
				}
			} else {
				// JSON 不完整，尝试提取部分内容
				raw := string(call.Arguments)
				if idx := strings.Index(raw, `"content"`); idx >= 0 {
					after := strings.TrimSpace(raw[idx+10:])
					if len(after) > 0 && after[0] == '"' {
						end := strings.LastIndex(after[1:], `"`)
						if end > 0 {
							var s string
							if json.Unmarshal([]byte(after[:end+2]), &s) == nil {
								synBuf.WriteString(s)
							}
						}
					}
				}
			}
		} else {
			realCalls = append(realCalls, call)
		}
	}
	return synBuf.String(), realCalls
}

func isCutShort(reason string) bool {
	return cutShortReasons[strings.ToLower(strings.TrimSpace(reason))]
}

func (s *server) handleAntiTruncation(
	w http.ResponseWriter, r *http.Request,
	request chatRequest, realModel string,
) {
	requestID := newID("chatcmpl")
	log.Printf("[anti-trunc] 开始处理 model=%s requestID=%s", realModel, requestID)
	request.Model = realModel
	created := time.Now().Unix()
	displayModel := variantPrefixAntiTrunc + realModel

	if err := streamHeaders(w); err != nil {
		return
	}

	type workResult struct {
		collectedText string
		lastResult    *generationResult
		realCalls     []aistudio.FunctionCall
		err           error
	}

	done := make(chan workResult, 1)
	var writing sync.Mutex

	go func() {
		var collected strings.Builder
		var lastResult *generationResult
		var lastRealCalls []aistudio.FunctionCall

		// 检测用户原始请求是否带有 function calling 工具
		userHasTools := hasUserTools(request.Tools)
		if userHasTools {
			log.Printf("[anti-trunc] 用户请求已包含 function tools，使用 system prompt 续写策略")
		}

		originalMessages := make([]chatMessage, len(request.Messages))
		copy(originalMessages, request.Messages)

		for continuation := 0; continuation < maxContinuationAttempts; continuation++ {
			toolName := randomToolName()

			// 构建当前轮请求
			reqCopy := request
			reqCopy.Messages = make([]chatMessage, len(originalMessages))
			copy(reqCopy.Messages, originalMessages)

			if continuation > 0 {
				currentText := collected.String()
				assistantContent, _ := json.Marshal(currentText)
				continuationPrompt, _ := json.Marshal(buildContinuationPrompt(toolName, currentText))
				reqCopy.Messages = append(reqCopy.Messages,
					chatMessage{Role: "assistant", Content: assistantContent},
					chatMessage{Role: "user", Content: continuationPrompt},
				)
			}

			// 注入合成工具（仅在用户无自定义 tools 时使用工具注入策略）
			if !userHasTools {
				injectAntiTruncTools(&reqCopy, toolName)
			}

			// 注入系统指令
			var controlText string
			if userHasTools {
				// 用户有 tools 时用纯 system prompt 指令，不注入合成工具
				controlText = buildPromptOnlyInstruction(continuation > 0)
			} else {
				controlText = buildControlInstruction(toolName, continuation > 0)
			}
			controlInstruction, _ := json.Marshal(controlText)
			reqCopy.Messages = append([]chatMessage{
				{Role: "system", Content: controlInstruction},
			}, reqCopy.Messages...)

			genReq, err := reqCopy.toGenerateRequest(requestID)
			if err != nil {
				// 首轮构建失败是致命错误，续传轮次构建失败则 fallback 返回已有内容
				if continuation == 0 {
					done <- workResult{err: err}
				} else {
					log.Printf("[anti-trunc] 续传第 %d 轮请求构建失败，返回已有内容: %v", continuation, err)
					done <- workResult{collectedText: collected.String(), lastResult: lastResult, realCalls: lastRealCalls}
				}
				return
			}
			if err := s.restoreMediaLinks(genReq.Contents); err != nil {
				if continuation == 0 {
					done <- workResult{err: err}
				} else {
					log.Printf("[anti-trunc] 续传第 %d 轮 restoreMediaLinks 失败，返回已有内容: %v", continuation, err)
					done <- workResult{collectedText: collected.String(), lastResult: lastResult, realCalls: lastRealCalls}
				}
				return
			}
			if err := inlineRemoteMedia(r.Context(), genReq.Contents); err != nil {
				if continuation == 0 {
					done <- workResult{err: err}
				} else {
					log.Printf("[anti-trunc] 续传第 %d 轮 inlineRemoteMedia 失败，返回已有内容: %v", continuation, err)
					done <- workResult{collectedText: collected.String(), lastResult: lastResult, realCalls: lastRealCalls}
				}
				return
			}
			s.thoughtSignatures.Restore(genReq.Contents)
			genReq.Unary = false

			events, err := s.service.Generate(r.Context(), genReq)
			if err != nil {
				// 续传轮次的 Generate 失败时 fallback
				if continuation == 0 {
					done <- workResult{err: err}
				} else {
					log.Printf("[anti-trunc] 续传第 %d 轮 Generate 失败，返回已有内容: %v", continuation, err)
					done <- workResult{collectedText: collected.String(), lastResult: lastResult, realCalls: lastRealCalls}
				}
				return
			}

			result, err := consumeEvents(r.Context(), events, nil)
			if err != nil {
				if continuation == 0 {
					done <- workResult{err: err}
				} else {
					log.Printf("[anti-trunc] 续传第 %d 轮 consumeEvents 失败，返回已有内容: %v", continuation, err)
					done <- workResult{collectedText: collected.String(), lastResult: lastResult, realCalls: lastRealCalls}
				}
				return
			}

			lastResult = &result
			var newContent string
			if !userHasTools {
				var realCalls []aistudio.FunctionCall
				newContent, realCalls = extractSyntheticContent(&result, toolName)
				lastRealCalls = realCalls
			} else {
				// 纯 prompt 模式，直接取 text 输出
				newContent = result.text.String()
				lastRealCalls = result.toolCalls
			}

			if newContent != "" {
				collected.WriteString(newContent)
			} else if text := result.text.String(); text != "" && !userHasTools {
				// 合成工具未被调用，fallback 到 text
				collected.WriteString(text)
			}

			log.Printf("[anti-trunc] 第 %d 轮完成，finishReason=%s，累计长度=%d",
				continuation, result.finishReason, collected.Len())

			if !isCutShort(result.finishReason) || collected.Len() == 0 {
				break
			}
		}

		done <- workResult{
			collectedText: collected.String(),
			lastResult:    lastResult,
			realCalls:     lastRealCalls,
		}
	}()

	// 心跳循环
	ticker := time.NewTicker(variantHeartbeatInterval)
	defer ticker.Stop()

	var wr workResult
waitLoop:
	for {
		select {
		case wr = <-done:
			break waitLoop
		case <-ticker.C:
			writing.Lock()
			err := writeAntiTruncHeartbeat(w, requestID, created, displayModel)
			writing.Unlock()
			if err != nil {
				return
			}
		case <-r.Context().Done():
			return
		}
	}

	if wr.err != nil {
		writeVariantError(w, wr.err)
		return
	}

	if wr.lastResult == nil {
		_ = writeSSE(w, "", map[string]any{"error": map[string]any{
			"message": "anti-truncation: no result",
			"type":    "server_error",
		}})
		return
	}

	// 用合成收集的文本替换 result 正文，并移除合成工具调用
	if wr.collectedText != "" {
		wr.lastResult.text.Reset()
		wr.lastResult.text.WriteString(wr.collectedText)
		wr.lastResult.toolCalls = wr.realCalls
	}
	// 抗截断完成后强制 finish 为 stop
	wr.lastResult.finishReason = "stop"
	log.Printf("[anti-trunc] 完成 model=%s, 最终文本长度=%d, 真实工具调用=%d",
		realModel, len(wr.collectedText), len(wr.lastResult.toolCalls))
	emitFullResult(w, requestID, created, displayModel, wr.lastResult, request.Tools, request.StreamOptions.IncludeUsage)
}

// buildPromptOnlyInstruction 用户有自定义 tools 时的纯 system prompt 续写指令（不注入合成工具）
func buildPromptOnlyInstruction(continuation bool) string {
	if continuation {
		return "Your previous reply was truncated. Continue outputting all remaining content " +
			"directly. Do not repeat what was already output. Continue from the exact point " +
			"where the previous reply ended. Preserve language, formatting, and any unfinished " +
			"sentence or markup."
	}
	return "Provide a complete and thorough response. If the response is long, ensure you " +
		"cover all requested content without omitting any sections."
}

// ---------- 公共辅助 ----------

// emitFullResult 把完整结果作为 SSE 流发送（role + reasoning + content + tools + finish + usage + DONE）
func emitFullResult(w http.ResponseWriter, id string, created int64, model string, result *generationResult, tools []openAITool, includeUsage bool) {
	// role
	_ = writeChatChunk(w, id, created, model, 0,
		map[string]any{"role": "assistant"}, nil, false)

	// 思考
	if reasoning := result.reasoning.String(); reasoning != "" {
		_ = writeChatChunk(w, id, created, model, 0,
			map[string]any{"reasoning_content": reasoning}, nil, false)
	}

	// 正文
	if text := result.text.String(); text != "" {
		_ = writeChatChunk(w, id, created, model, 0,
			map[string]any{"content": text}, nil, false)
	}

	// 工具调用
	for i, call := range result.toolCalls {
		_ = writeChatChunk(w, id, created, model, 0,
			map[string]any{"tool_calls": []map[string]any{{
				"index":    i,
				"id":       call.ID,
				"type":     "function",
				"function": map[string]any{"name": call.Name, "arguments": string(call.Arguments)},
			}}}, nil, false)
	}

	// 结束
	finish := chatFinishReason(*result, tools)
	_ = writeChatChunk(w, id, created, model, 0,
		map[string]any{}, &finish, false)

	// usage
	if includeUsage && result.usage != nil {
		_ = writeSSE(w, "", map[string]any{
			"id": id, "object": "chat.completion.chunk",
			"created": created, "model": model,
			"choices": []any{}, "usage": openAIUsage(result.usage),
		})
	}
	_ = writeSSEText(w, "[DONE]")
}

// writeVariantError 写入 SSE 错误事件
func writeVariantError(w http.ResponseWriter, err error) {
	status := statusFromError(err)
	code := openAIErrorCode(err)
	_ = writeSSE(w, "", map[string]any{"error": map[string]any{
		"message": publicErrorMessage(status, err.Error()),
		"type":    openAIErrorType(status, code),
		"param":   nil,
		"code":    code,
	}})
}
