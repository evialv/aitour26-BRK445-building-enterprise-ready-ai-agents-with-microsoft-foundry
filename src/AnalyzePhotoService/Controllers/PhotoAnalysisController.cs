using Microsoft.Agents.AI;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.AI;
using SharedEntities;
using System.Text.Json;
using ZavaAgentsMetadata;
using ZavaMAFLocal;

namespace AnalyzePhotoService.Controllers;

[ApiController]
[Route("api/[controller]")]
public class PhotoAnalysisController : ControllerBase
{
    private readonly ILogger<PhotoAnalysisController> _logger;
    private readonly AIAgent _agentFxAgent;
    private readonly IChatClient _chatClient;

    public PhotoAnalysisController(
        ILogger<PhotoAnalysisController> logger,
        MAFLocalAgentProvider localAgentProvider,
        IChatClient chatClient)
    {
        _logger = logger;
        _agentFxAgent = localAgentProvider.GetAgentByName(AgentMetadata.GetAgentName(AgentType.PhotoAnalyzerAgent));
        _chatClient = chatClient;
    }

    [HttpPost("analyzellm")]
    public async Task<ActionResult<PhotoAnalysisResult>> AnalyzeLLMAsync([FromForm] IFormFile image, [FromForm] string prompt, CancellationToken cancellationToken = default)
    {
        if (image is null)
        {
            return BadRequest("No image file was provided.");
        }

        _logger.LogInformation($"{AgentMetadata.LogPrefixes.Llm} Analyzing photo. Prompt: {{Prompt}}", prompt);

        // LLM endpoint uses IChatClient with vision model per architecture doc
        return await AnalyzeWithVisionAsync(image, prompt, AgentMetadata.LogPrefixes.Llm, cancellationToken);
    }

    [HttpPost("analyzemaf_local")]  // Using constant AgentMetadata.FrameworkIdentifiers.MafLocal
    public async Task<ActionResult<PhotoAnalysisResult>> AnalyzeMAFLocalAsync([FromForm] IFormFile image, [FromForm] string prompt, CancellationToken cancellationToken = default)
    {
        if (image is null)
        {
            return BadRequest("No image file was provided.");
        }

        _logger.LogInformation($"{AgentMetadata.LogPrefixes.MafLocal} Analyzing photo. Prompt: {{Prompt}}", prompt);

        return await AnalyzeWithVisionAsync(image, prompt, AgentMetadata.LogPrefixes.MafLocal, cancellationToken);
    }

    [HttpPost("analyzemaf_foundry")]  // Using constant AgentMetadata.FrameworkIdentifiers.MafFoundry
    public async Task<ActionResult<PhotoAnalysisResult>> AnalyzeMAFFoundryAsync([FromForm] IFormFile image, [FromForm] string prompt, CancellationToken cancellationToken = default)
    {
        if (image is null)
        {
            return BadRequest("No image file was provided.");
        }

        _logger.LogInformation($"{AgentMetadata.LogPrefixes.MafFoundry} Analyzing photo. Prompt: {{Prompt}}", prompt);

        return await AnalyzeWithVisionAsync(image, prompt, AgentMetadata.LogPrefixes.MafFoundry, cancellationToken);
    }

    [HttpPost("analyzedirectcall")]
    public async Task<ActionResult<PhotoAnalysisResult>> AnalyzeDirectCallAsync([FromForm] IFormFile image, [FromForm] string prompt, CancellationToken cancellationToken = default)
    {
        if (image is null)
        {
            return BadRequest("No image file was provided.");
        }

        _logger.LogInformation("[DirectCall] Analyzing photo. Prompt: {Prompt}", prompt);

        // add a sleep of 3 seconds to emulate the image analysis time
        await Task.Delay(3000);

        // Fallback path.
        var fallbackDescription = BuildFallbackDescription(prompt);
        var fallback = new PhotoAnalysisResult
        {
            Description = fallbackDescription,
            DetectedMaterials = DetermineDetectedMaterials(prompt, image.FileName)
        };
        return Ok(fallback);
    }

    /// <summary>
    /// Analyzes image using Azure OpenAI vision model via IChatClient per architecture doc.
    /// Sends the actual image data as multimodal content.
    /// </summary>
    private async Task<ActionResult<PhotoAnalysisResult>> AnalyzeWithVisionAsync(
        IFormFile image,
        string userPrompt,
        string logPrefix,
        CancellationToken cancellationToken)
    {
        var fallbackDescription = BuildFallbackDescription(userPrompt);

        try
        {
            // Read image bytes
            using var ms = new MemoryStream();
            await image.CopyToAsync(ms, cancellationToken);
            var imageBytes = ms.ToArray();

            // Build multimodal chat message with image + text
            var systemPrompt = @"You are an AI assistant that analyzes photos of rooms for renovation and home-improvement projects.
Analyze the provided image and the user's prompt, then return a JSON object with exactly two fields:
  - description: a brief natural-language description of what the image shows and what renovation tasks are likely required
  - detectedMaterials: an array of short strings naming materials, finishes or items that appear relevant (e.g. 'paint', 'tile', 'wood', 'grout')

Return only valid JSON. Do not include any surrounding markdown or explanatory text.";

            var messages = new List<ChatMessage>
            {
                new(ChatRole.System, systemPrompt),
                new(ChatRole.User, 
                [
                    new DataContent(imageBytes, image.ContentType),
                    new TextContent($"ImageFileName: {image.FileName}\nUserPrompt: {userPrompt}")
                ])
            };

            var response = await _chatClient.GetResponseAsync(messages, cancellationToken: cancellationToken);
            var agentRawResponse = response.Text ?? string.Empty;

            _logger.LogInformation("{Prefix} Raw vision response length: {Length}", logPrefix, agentRawResponse.Length);

            if (TryParsePhotoAnalysis(agentRawResponse, out var parsed))
            {
                return Ok(parsed);
            }

            _logger.LogWarning("{Prefix} Parsed result invalid or incomplete. Using heuristic fallback. Raw: {Raw}", logPrefix, TrimForLog(agentRawResponse));
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "{Prefix} Vision analysis failed. Using heuristic fallback.", logPrefix);
        }

        // Fallback path.
        var fallback = new PhotoAnalysisResult
        {
            Description = fallbackDescription,
            DetectedMaterials = DetermineDetectedMaterials(userPrompt, image.FileName)
        };
        return Ok(fallback);
    }

    // Shared high-level analysis routine for text-only endpoints.
    private async Task<ActionResult<PhotoAnalysisResult>> AnalyzeWithAgentAsync(
        string userPrompt,
        string fileName,
        Func<string, Task<string>> invokeAgent,
        string logPrefix,
        CancellationToken cancellationToken)
    {
        var analysisPrompt = BuildAnalysisPrompt(userPrompt, fileName);
        var fallbackDescription = BuildFallbackDescription(userPrompt);

        try
        {
            var agentRawResponse = await invokeAgent(analysisPrompt);
            _logger.LogInformation("{Prefix} Raw agent response length: {Length}", logPrefix, agentRawResponse.Length);

            if (TryParsePhotoAnalysis(agentRawResponse, out var parsed))
            {
                return Ok(parsed);
            }

            _logger.LogWarning("{Prefix} Parsed result invalid or incomplete. Using heuristic fallback. Raw: {Raw}", logPrefix, TrimForLog(agentRawResponse));
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "{Prefix} Invocation failed. Using heuristic fallback.", logPrefix);
        }

        // Fallback path.
        var fallback = new PhotoAnalysisResult
        {
            Description = fallbackDescription,
            DetectedMaterials = DetermineDetectedMaterials(userPrompt, fileName)
        };
        return Ok(fallback);
    }

    // Agent invocation helper
    private async Task<string> GetAgentFxResponseAsync(string prompt)
    {
        var thread = _agentFxAgent.GetNewThread();
        var response = await _agentFxAgent.RunAsync(prompt, thread);
        return response?.Text ?? string.Empty;
    }

    private string BuildAnalysisPrompt(string prompt, string fileName)
    {
        return $@"You are an AI assistant that analyzes photos of rooms for renovation and home-improvement projects.
Given the image filename and the user's short prompt, return a JSON object with exactly two fields:
  - description: a brief natural-language description of what the image shows and what renovation tasks are likely required
  - detectedMaterials: an array of short strings naming materials, finishes or items that appear relevant (e.g. 'paint', 'tile', 'wood', 'grout')

Return only valid JSON. Do not include any surrounding markdown or explanatory text.

ImageFileName: {fileName}
UserPrompt: {prompt}
";
    }

    #region JSON parsing logic centralized.
    private bool TryParsePhotoAnalysis(string agentResponse, out PhotoAnalysisResult result)
    {
        result = default!;
        if (string.IsNullOrWhiteSpace(agentResponse)) return false;

        var json = ExtractJson(agentResponse);
        if (json is null) return false;

        try
        {
            using var doc = JsonDocument.Parse(json);
            var root = doc.RootElement;

            if (!root.TryGetProperty("description", out var descProp) || descProp.ValueKind != JsonValueKind.String)
                return false;

            if (!root.TryGetProperty("detectedMaterials", out var materialsProp) || materialsProp.ValueKind != JsonValueKind.Array)
                return false;

            var description = descProp.GetString() ?? string.Empty;
            var materials = new List<string>();
            foreach (var item in materialsProp.EnumerateArray())
            {
                if (item.ValueKind == JsonValueKind.String && !string.IsNullOrWhiteSpace(item.GetString()))
                {
                    materials.Add(item.GetString()!);
                }
            }

            if (string.IsNullOrWhiteSpace(description) || materials.Count == 0)
                return false;

            result = new PhotoAnalysisResult
            {
                Description = description,
                DetectedMaterials = materials.ToArray()
            };
            return true;
        }
        catch (JsonException)
        {
            return false;
        }
    }

    // Extracts first valid JSON object substring from a response.
    private static string? ExtractJson(string raw)
    {
        var first = raw.IndexOf('{');
        var last = raw.LastIndexOf('}');
        if (first < 0 || last <= first) return null;
        return raw.Substring(first, last - first + 1);
    }
    #endregion

    #region Fallback description and logging
    private static string BuildFallbackDescription(string prompt) =>
        $"Photo analysis for prompt: '{prompt}'. Detected a room that needs renovation work. The image shows surfaces that require preparation and finishing.";

    private static string TrimForLog(string value, int max = 500)
        => value.Length <= max ? value : value.Substring(0, max) + "...";

    // Simple heuristic detector.
    private string[] DetermineDetectedMaterials(string prompt, string? fileName)
    {
        var materials = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        var promptLower = prompt.ToLowerInvariant();
        var fileNameLower = fileName?.ToLowerInvariant() ?? string.Empty;

        bool Contains(params string[] keys) => keys.Any(k => promptLower.Contains(k) || fileNameLower.Contains(k));

        if (Contains("paint", "wall"))
            AddRange(materials, "paint", "wall", "surface preparation");

        if (Contains("wood", "deck"))
            AddRange(materials, "wood", "stain", "sanding");

        if (Contains("tile", "bathroom"))
            AddRange(materials, "tile", "grout", "adhesive");

        if (Contains("garden", "landscape"))
            AddRange(materials, "soil", "plants", "tools");

        if (materials.Count == 0)
            AddRange(materials, "general tools", "measuring", "safety equipment");

        return materials.ToArray();
    }

    private static void AddRange(HashSet<string> set, params string[] values)
    {
        foreach (var v in values)
        {
            if (!string.IsNullOrWhiteSpace(v)) set.Add(v);
        }
    }

    // Simple DTO used to deserialize the AI's JSON response (kept for potential future direct deserialization).
    private record AiPhotoAnalysisResult(string Description, string[] DetectedMaterials);
    #endregion
}
