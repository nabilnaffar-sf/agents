/*
 * Copyright 2026, Salesforce, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import { Connection, Logger, PollingClient, SfError } from '@salesforce/core';
import { Duration } from '@salesforce/kit';

// ====================================================
//               STDM Session Types
// ====================================================

export type StdmStep = {
  id: string;
  type: 'LLM_STEP' | 'ACTION_STEP' | 'VARIABLE_UPDATE_STEP' | 'TRUST_GUARDRAILS_STEP' | (string & NonNullable<unknown>);
  name?: string;
  startOffsetMs?: number;
  durationMs?: number;
  input?: Record<string, unknown>;
  output?: Record<string, unknown>;
  llmOutput?: string;
  model?: string;
  finishReason?: string;
};

export type StdmInteraction = {
  id: string;
  type?: string;
  topic?: string;
  traceId?: string;
  startOffsetMs?: number;
  durationMs?: number;
  steps: StdmStep[];
};

export type StdmSessionData = {
  sessionId: string;
  channel?: string;
  startTime?: string;
  endTime?: string;
  durationMs?: number;
  turns: StdmInteraction[];
};

// ====================================================
//               Scorer Run Types
// ====================================================

export type ScorerRunStatus = 'New' | 'InProgress' | 'Completed' | 'Failed';

export type BulkScoringTriggerResponse = {
  actionName: string;
  errors: string[] | null;
  isSuccess: boolean;
  outputValues: { jobId?: string } | null;
};

export type ScorerRunJobStatus = {
  status: ScorerRunStatus;
  jobId?: string;
};

export type ScorerRunResult = {
  sessionId: string;
  scorerName: string;
  value: string | null;
  error?: string;
};

export type ScorerRunResponse = {
  status: ScorerRunStatus;
  results: ScorerRunResult[];
};

export type ScorerRunOptions = {
  timeout?: Duration;
  pollInterval?: number;
  maxRetries?: number;
};

// ====================================================
//               STDM Query Results
// ====================================================

type StdmSessionRecord = {
  ssot__Id__c: string;
  ssot__StartTimestamp__c?: string;
  ssot__EndTimestamp__c?: string;
  ssot__AiAgentChannelType__c?: string;
};

type StdmInteractionRecord = {
  ssot__Id__c: string;
  ssot__AiAgentSessionId__c: string;
  ssot__StartTimestamp__c?: string;
  ssot__EndTimestamp__c?: string;
  ssot__AiAgentTopic__c?: string;
};

type StdmStepRecord = {
  ssot__Id__c: string;
  ssot__AiAgentInteractionId__c: string;
  ssot__StepType__c?: string;
  ssot__StepName__c?: string;
  ssot__StartTimestamp__c?: string;
  ssot__EndTimestamp__c?: string;
  ssot__Input__c?: string;
  ssot__Output__c?: string;
};

// ====================================================
//               Scorer Definition Query Types
// ====================================================

type ScorerDefinitionRecord = {
  Id: string;
  DeveloperName: string;
  MasterLabel: string;
  EngineType: string;
  Status: string;
  DataType: string;
  PromptTemplateRef?: string;
};

type PromptTemplateRecord = {
  Id: string;
  DeveloperName: string;
  IsActive: boolean;
};

// ====================================================
//               ScorerRunner
// ====================================================

const DEFAULT_MAX_RETRIES = 3;
const RETRY_BACKOFF_MS = 1000;

export class ScorerRunner {
  private connection: Connection;
  private logger: Logger;
  private apiVersion: string;

  public constructor(connection: Connection, options?: { apiVersion?: string }) {
    this.connection = connection;
    this.logger = Logger.childFromRoot('ScorerRunner');
    this.apiVersion = options?.apiVersion ?? '66.0';
  }

  /**
   * Run a scorer against live session IDs by triggering the bulk scoring API.
   * The sessions are fetched from production STDM by the platform.
   * Supports partial success — sessions that fail after retries are reported
   * in the results with an error field rather than throwing.
   *
   * @param scorerName - The API name of the scorer definition
   * @param sessionIds - List of session UUIDs to score
   * @param options - Optional polling/retry configuration
   */
  public async run(
    scorerName: string,
    sessionIds: string[],
    options?: ScorerRunOptions
  ): Promise<ScorerRunResponse> {
    if (!sessionIds.length) {
      return { status: 'Completed', results: [] };
    }

    await this.validateScorer(scorerName);

    const maxRetries = options?.maxRetries ?? DEFAULT_MAX_RETRIES;

    this.logger.debug(`Triggering bulk scoring for scorer '${scorerName}' on ${sessionIds.length} sessions`);

    const triggerResponse = await this.withRetry(
      () => this.triggerBulkScoring(scorerName, sessionIds),
      maxRetries,
      'triggerBulkScoring'
    );

    if (!triggerResponse.isSuccess) {
      const errorMsg = triggerResponse.errors?.join('; ') ?? 'Unknown error triggering bulk scoring';
      throw SfError.create({
        name: 'ScorerRunTriggerError',
        message: `Failed to trigger bulk scoring for '${scorerName}': ${errorMsg}`,
      });
    }

    const jobId = triggerResponse.outputValues?.jobId;
    this.logger.debug(`Bulk scoring triggered, jobId: ${jobId ?? 'none'}`);

    const result = await this.pollForResults(scorerName, sessionIds, {
      timeout: options?.timeout ?? Duration.minutes(10),
      pollInterval: options?.pollInterval ?? 5000,
    });

    return result;
  }

  /**
   * Run a scorer against provided mock session data without querying STDM.
   * This calls the scorer's prompt template evaluation endpoint with the session
   * payload injected directly. Each session is evaluated independently — failures
   * on individual sessions do not abort the entire run.
   *
   * @param scorerName - The API name of the scorer definition
   * @param sessionData - List of session JSON payloads to evaluate
   * @param options - Optional retry configuration
   */
  public async runWithMockData(
    scorerName: string,
    sessionData: StdmSessionData[],
    options?: Pick<ScorerRunOptions, 'maxRetries'>
  ): Promise<ScorerRunResponse> {
    if (!sessionData.length) {
      return { status: 'Completed', results: [] };
    }

    await this.validateScorer(scorerName);

    const maxRetries = options?.maxRetries ?? DEFAULT_MAX_RETRIES;

    this.logger.debug(`Running scorer '${scorerName}' with ${sessionData.length} mock sessions`);

    const results: ScorerRunResult[] = [];

    for (const session of sessionData) {
      // eslint-disable-next-line no-await-in-loop
      const result = await this.evaluateWithRetry(scorerName, session, maxRetries);
      results.push(result);
    }

    const hasErrors = results.some((r) => r.error);
    const allFailed = results.every((r) => r.error);
    let status: ScorerRunStatus;
    if (allFailed) {
      status = 'Failed';
    } else if (hasErrors) {
      status = 'Completed';
    } else {
      status = 'Completed';
    }

    return { status, results };
  }

  /**
   * Fetch session data from STDM for the given session IDs.
   * Sessions that cannot be found are omitted from the result (not thrown).
   */
  public async fetchSessionData(
    sessionIds: string[],
    options?: Pick<ScorerRunOptions, 'maxRetries'>
  ): Promise<StdmSessionData[]> {
    const maxRetries = options?.maxRetries ?? DEFAULT_MAX_RETRIES;
    const sessions: StdmSessionData[] = [];

    for (const sessionId of sessionIds) {
      // eslint-disable-next-line no-await-in-loop
      const session = await this.withRetry(
        () => this.querySessionFromStdm(sessionId),
        maxRetries,
        `fetchSession:${sessionId}`
      );
      if (session) {
        sessions.push(session);
      }
    }

    return sessions;
  }

  // ====================================================
  //               Private Methods
  // ====================================================

  private async validateScorer(scorerName: string): Promise<void> {
    const escapedName = scorerName.replace(/'/g, "''");

    let records: ScorerDefinitionRecord[];
    try {
      const result = await this.connection.query<ScorerDefinitionRecord>(
        `SELECT Id, DeveloperName, MasterLabel, EngineType, Status, DataType
         FROM AiAgentScorerDefinition
         WHERE DeveloperName = '${escapedName}' LIMIT 1`
      );
      records = result.records;
    } catch {
      this.logger.debug('AiAgentScorerDefinition not queryable, skipping validation');
      return;
    }

    if (!records.length) {
      throw SfError.create({
        name: 'ScorerNotFound',
        message: `Scorer '${scorerName}' not found. Verify the DeveloperName exists in AiAgentScorerDefinition.`,
        actions: [
          'Run `sf agent scorer list` to see available scorers in this org.',
          'Ensure the scorer metadata has been deployed to the target org.',
        ],
      });
    }

    const scorer = records[0];

    if (scorer.EngineType !== 'PromptTemplate') {
      throw SfError.create({
        name: 'ScorerEngineTypeNotSupported',
        message: `Scorer '${scorerName}' has engine type '${scorer.EngineType}' — only 'PromptTemplate' is supported for run.`,
        actions: [
          'Manual scorers cannot be executed programmatically.',
          'Create a scorer with engineType: PromptTemplate to use this API.',
        ],
      });
    }

    // Scorer status Draft or Available is fine — but the prompt template must be active.
    await this.validatePromptTemplateActive(scorerName, escapedName);
  }

  private async validatePromptTemplateActive(scorerName: string, escapedName: string): Promise<void> {
    try {
      const result = await this.connection.query<PromptTemplateRecord>(
        `SELECT Id, DeveloperName, IsActive
         FROM GenAiPromptTemplate
         WHERE DeveloperName = '${escapedName}' LIMIT 1`
      );

      if (!result.records.length) {
        throw SfError.create({
          name: 'ScorerPromptTemplateNotFound',
          message: `Prompt template for scorer '${scorerName}' not found.`,
          actions: [
            'Ensure the GenAiPromptTemplate metadata is deployed alongside the scorer definition.',
          ],
        });
      }

      const template = result.records[0];
      if (!template.IsActive) {
        throw SfError.create({
          name: 'ScorerPromptTemplateInactive',
          message: `Prompt template '${template.DeveloperName}' for scorer '${scorerName}' is inactive.`,
          actions: [
            'Activate the prompt template in Setup > Prompt Builder before running the scorer.',
            'Scorers with inactive prompt templates will fail at execution time.',
          ],
        });
      }
    } catch (error) {
      if (error instanceof SfError && (error.name === 'ScorerPromptTemplateNotFound' || error.name === 'ScorerPromptTemplateInactive')) {
        throw error;
      }
      this.logger.debug('GenAiPromptTemplate not queryable, skipping prompt template validation');
    }
  }

  private async withRetry<T>(
    fn: () => Promise<T>,
    maxRetries: number,
    label: string
  ): Promise<T> {
    let lastError: unknown;
    for (let attempt = 0; attempt <= maxRetries; attempt++) {
      try {
        // eslint-disable-next-line no-await-in-loop
        return await fn();
      } catch (error) {
        lastError = error;
        if (attempt < maxRetries) {
          const delay = RETRY_BACKOFF_MS * Math.pow(2, attempt);
          this.logger.debug(`${label} attempt ${attempt + 1} failed, retrying in ${delay}ms`);
          // eslint-disable-next-line no-await-in-loop
          await this.sleep(delay);
        }
      }
    }
    throw lastError;
  }

  private async evaluateWithRetry(
    scorerName: string,
    session: StdmSessionData,
    maxRetries: number
  ): Promise<ScorerRunResult> {
    let lastError: string | undefined;

    for (let attempt = 0; attempt <= maxRetries; attempt++) {
      try {
        // eslint-disable-next-line no-await-in-loop
        const result = await this.evaluateScorerWithData(scorerName, session);
        if (!result.error) {
          return result;
        }
        lastError = result.error;
        if (attempt < maxRetries) {
          const delay = RETRY_BACKOFF_MS * Math.pow(2, attempt);
          this.logger.debug(
            `Scorer evaluation for session ${session.sessionId} returned error, retrying in ${delay}ms (attempt ${attempt + 1}/${maxRetries})`
          );
          // eslint-disable-next-line no-await-in-loop
          await this.sleep(delay);
        }
      } catch (error) {
        lastError = error instanceof Error ? error.message : String(error);
        if (attempt < maxRetries) {
          const delay = RETRY_BACKOFF_MS * Math.pow(2, attempt);
          this.logger.debug(
            `Scorer evaluation for session ${session.sessionId} threw, retrying in ${delay}ms (attempt ${attempt + 1}/${maxRetries})`
          );
          // eslint-disable-next-line no-await-in-loop
          await this.sleep(delay);
        }
      }
    }

    return {
      sessionId: session.sessionId,
      scorerName,
      value: null,
      error: lastError ?? `Failed after ${maxRetries + 1} attempts`,
    };
  }

  private async triggerBulkScoring(
    scorerName: string,
    sessionIds: string[]
  ): Promise<BulkScoringTriggerResponse> {
    const url = `/services/data/v${this.apiVersion}/actions/standard/triggerAgentBulkScoring`;

    const payload = {
      inputs: [
        {
          inputScope: 'Session',
          scorerApiNames: [scorerName],
          inputIds: sessionIds,
        },
      ],
    };

    const responses = await this.connection.request<BulkScoringTriggerResponse[]>({
      method: 'POST',
      url: `${this.connection.instanceUrl}${url}`,
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    if (!responses || responses.length === 0) {
      throw SfError.create({
        name: 'ScorerRunTriggerError',
        message: 'Empty response from triggerAgentBulkScoring',
      });
    }

    return responses[0];
  }

  private async pollForResults(
    scorerName: string,
    sessionIds: string[],
    options: { timeout: Duration; pollInterval: number }
  ): Promise<ScorerRunResponse> {
    const client = await PollingClient.create({
      poll: async (): Promise<{ completed: boolean; payload?: ScorerRunResponse }> => {
        const results = await this.queryScorerResults(scorerName, sessionIds);

        if (results.length >= sessionIds.length) {
          return {
            completed: true,
            payload: { status: 'Completed', results },
          };
        }

        return { completed: false };
      },
      frequency: Duration.milliseconds(options.pollInterval),
      timeout: options.timeout,
    });

    try {
      return await client.subscribe<ScorerRunResponse>();
    } catch {
      this.logger.debug('Polling timed out, returning partial results');
      const partialResults = await this.queryScorerResults(scorerName, sessionIds);
      const missingSessions = sessionIds.filter(
        (id) => !partialResults.some((r) => r.sessionId === id)
      );
      const errorResults: ScorerRunResult[] = missingSessions.map((id) => ({
        sessionId: id,
        scorerName,
        value: null,
        error: 'Scoring timed out — result not available yet',
      }));
      return {
        status: partialResults.length > 0 ? 'Completed' : 'Failed',
        results: [...partialResults, ...errorResults],
      };
    }
  }

  private async queryScorerResults(scorerName: string, sessionIds: string[]): Promise<ScorerRunResult[]> {
    const sessionIdList = sessionIds.map((id) => `'${id}'`).join(',');

    const tagQuery = `
      SELECT ssot__Id__c, ssot__AiAgentTagValue__c, ssot__AiAgentSessionId__c
      FROM ssot__AiAgentTag__dlm
      WHERE ssot__AiAgentSessionId__c IN (${sessionIdList})
        AND ssot__AiAgentTagDefinitionId__c IN (
          SELECT ssot__Id__c FROM ssot__AiAgentTagDefinition__dlm
          WHERE ssot__DeveloperName__c = '${scorerName}'
        )
    `.trim();

    try {
      const queryResult = await this.connection.query<{
        ssot__Id__c: string;
        ssot__AiAgentTagValue__c: string;
        ssot__AiAgentSessionId__c: string;
      }>(tagQuery);

      return queryResult.records.map((record) => ({
        sessionId: record.ssot__AiAgentSessionId__c,
        scorerName,
        value: record.ssot__AiAgentTagValue__c,
      }));
    } catch {
      return [];
    }
  }

  private async evaluateScorerWithData(
    scorerName: string,
    sessionData: StdmSessionData
  ): Promise<ScorerRunResult> {
    const url = `/services/data/v${this.apiVersion}/einstein/ai-scoring/score`;

    const payload = {
      scorerApiName: scorerName,
      inputScope: 'Session',
      sessionData: {
        sessionId: sessionData.sessionId,
        channel: sessionData.channel,
        startTime: sessionData.startTime,
        endTime: sessionData.endTime,
        durationMs: sessionData.durationMs,
        turns: sessionData.turns.map((turn) => ({
          id: turn.id,
          type: turn.type,
          topic: turn.topic,
          steps: turn.steps.map((step) => ({
            id: step.id,
            type: step.type,
            name: step.name,
            input: step.input,
            output: step.output,
            llmOutput: step.llmOutput,
          })),
        })),
      },
    };

    const response = await this.connection.request<{
      value?: string;
      error?: string;
      isSuccess?: boolean;
    }>({
      method: 'POST',
      url: `${this.connection.instanceUrl}${url}`,
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    if (response.isSuccess === false || response.error) {
      return {
        sessionId: sessionData.sessionId,
        scorerName,
        value: null,
        error: response.error ?? 'Scorer evaluation failed',
      };
    }

    return {
      sessionId: sessionData.sessionId,
      scorerName,
      value: response.value ?? null,
    };
  }

  private async querySessionFromStdm(sessionId: string): Promise<StdmSessionData | null> {
    const sessionResult = await this.connection.query<StdmSessionRecord>(
      `SELECT ssot__Id__c, ssot__StartTimestamp__c, ssot__EndTimestamp__c, ssot__AiAgentChannelType__c
       FROM ssot__AiAgentSession__dlm
       WHERE ssot__Id__c = '${sessionId}' LIMIT 1`
    );

    if (!sessionResult.records.length) {
      this.logger.debug(`Session ${sessionId} not found in STDM`);
      return null;
    }

    const sessionRecord = sessionResult.records[0];

    const interactionResult = await this.connection.query<StdmInteractionRecord>(
      `SELECT ssot__Id__c, ssot__StartTimestamp__c, ssot__EndTimestamp__c, ssot__AiAgentTopic__c
       FROM ssot__AiAgentInteraction__dlm
       WHERE ssot__AiAgentSessionId__c = '${sessionId}'
       ORDER BY ssot__StartTimestamp__c ASC`
    );

    const interactions: StdmInteraction[] = [];

    for (const interaction of interactionResult.records) {
      // eslint-disable-next-line no-await-in-loop
      const stepsResult = await this.connection.query<StdmStepRecord>(
        `SELECT ssot__Id__c, ssot__StepType__c, ssot__StepName__c,
                ssot__StartTimestamp__c, ssot__EndTimestamp__c, ssot__Input__c, ssot__Output__c
         FROM ssot__AiAgentInteractionStep__dlm
         WHERE ssot__AiAgentInteractionId__c = '${interaction.ssot__Id__c}'
         ORDER BY ssot__StartTimestamp__c ASC`
      );

      const steps: StdmStep[] = stepsResult.records.map((step) => ({
        id: step.ssot__Id__c,
        type: step.ssot__StepType__c ?? 'UNKNOWN',
        name: step.ssot__StepName__c,
        input: step.ssot__Input__c ? (JSON.parse(step.ssot__Input__c) as Record<string, unknown>) : undefined,
        output: step.ssot__Output__c ? (JSON.parse(step.ssot__Output__c) as Record<string, unknown>) : undefined,
      }));

      interactions.push({
        id: interaction.ssot__Id__c,
        topic: interaction.ssot__AiAgentTopic__c,
        steps,
      });
    }

    const startTime = sessionRecord.ssot__StartTimestamp__c;
    const endTime = sessionRecord.ssot__EndTimestamp__c;
    let durationMs: number | undefined;
    if (startTime && endTime) {
      durationMs = new Date(endTime).getTime() - new Date(startTime).getTime();
    }

    return {
      sessionId: sessionRecord.ssot__Id__c,
      channel: sessionRecord.ssot__AiAgentChannelType__c,
      startTime,
      endTime,
      durationMs,
      turns: interactions,
    };
  }

  // eslint-disable-next-line class-methods-use-this
  private sleep(ms: number): Promise<void> {
    return new Promise((resolve) => {
      setTimeout(resolve, ms);
    });
  }
}
